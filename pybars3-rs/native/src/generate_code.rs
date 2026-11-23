//! `GenerateCode`, the native replacement of `pybars.Compiler._generate_code`.

use std::panic::catch_unwind;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use pybars3_compiler::{Error, Fallback, Options, PybarsVersion, UnicodeVersion};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::recursion::Recursion;

/// Why a call ran pybars' own `_generate_code`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reason {
    /// The port is disabled in this interpreter (see the Python package's
    /// `unavailable_reason()`): every call runs pybars.
    Unavailable,
    /// `pybars._compiler.debug` is set: pybars prints the tree.
    Debug,
    /// The source is not exactly a `str`: pybars raises `PybarsError`, or
    /// parses a `str` subclass through its own methods.
    NotStr,
    /// The compiler overrides one of the hooks that pybars' `_generate_code`
    /// calls (`whitespace_control` / `clean_whitespace`, `_extract_word`,
    /// `_handlebars`, `_compiler`, `__getattribute__`), or is not a
    /// `pybars.Compiler`.
    CustomCompiler,
    /// The source has a lone surrogate, which Rust strings cannot hold.
    Surrogates,
    /// An integer literal that `int()` rejects: pybars raises `ValueError`.
    IntConversion,
    /// Nesting too deep for the port, or for the stack headroom left below
    /// `sys.getrecursionlimit()`: pybars compiles it or raises
    /// `RecursionError`.
    NestingDepth,
    /// The port panicked: a bug in pybars3-rs, which pybars' result hides.
    Panic,
    /// The port returned an error or code that the binding does not
    /// expect: a bug in pybars3-rs, which pybars' result hides.
    Internal,
}

impl Reason {
    const ALL: [Reason; 9] = [
        Reason::Unavailable,
        Reason::Debug,
        Reason::NotStr,
        Reason::CustomCompiler,
        Reason::Surrogates,
        Reason::IntConversion,
        Reason::NestingDepth,
        Reason::Panic,
        Reason::Internal,
    ];

    /// The reason's key in `counters()`.
    const fn name(self) -> &'static str {
        match self {
            Reason::Unavailable => "unavailable",
            Reason::Debug => "debug",
            Reason::NotStr => "not_str",
            Reason::CustomCompiler => "custom_compiler",
            Reason::Surrogates => "surrogates",
            Reason::IntConversion => "int_conversion",
            Reason::NestingDepth => "nesting_depth",
            Reason::Panic => "panic",
            Reason::Internal => "internal",
        }
    }
}

impl From<Fallback> for Reason {
    fn from(fallback: Fallback) -> Self {
        match fallback {
            Fallback::IntConversion => Reason::IntConversion,
            Fallback::NestingDepth => Reason::NestingDepth,
            // `Fallback` is non-exhaustive: a new reason runs pybars too.
            _ => Reason::Internal,
        }
    }
}

/// The default hooks of `pybars.Compiler` that the fast path reproduces,
/// together with the verified release and Unicode database.
struct FastPath {
    pybars_version: PybarsVersion,
    unicode_version: UnicodeVersion,
    /// `pybars.Compiler`.
    compiler_class: Py<PyAny>,
    /// `object.__getattribute__`.
    object_getattribute: Py<PyAny>,
    /// `pybars.Compiler.whitespace_control` (0.9.7) or `clean_whitespace` (0.9.6).
    whitespace_hook: Py<PyAny>,
    /// `pybars.Compiler._handlebars`.
    handlebars: Py<PyAny>,
    /// `pybars.Compiler._extract_word`.
    extract_word: Py<PyAny>,
    /// `pybars.Compiler._compiler`.
    original_compile_grammar: Py<PyAny>,
    /// `pybars_rs.Compiler._compiler` descriptor.
    compile_grammar_descriptor: Py<PyAny>,
    /// Interned attribute names checked on the fast path.
    name_dict: Py<PyString>,
    name_mro: Py<PyString>,
    name_getattribute: Py<PyString>,
    name_whitespace_hook: Py<PyString>,
    name_handlebars: Py<PyString>,
    name_extract_word: Py<PyString>,
    name_compiler: Py<PyString>,
}

/// The objects of the installed pybars, and of the Python package, that
/// the fast path and the fallback need.
struct Pybars {
    /// `pybars._compiler`, for its `debug` flag.
    compiler_module: Py<PyAny>,
    /// `pybars._compiler.FunctionContainer`.
    function_container: Py<PyAny>,
    /// `pybars.Compiler._generate_code`.
    original: Py<PyAny>,
    /// `pybars.PybarsError`.
    error_class: Py<PyAny>,
    /// The `contextvars.ContextVar` holding the current fallback's grammar
    /// slot (`[grammar_or_None]`) that `pybars_rs.Compiler._compiler` reads.
    fallback_grammar: Py<PyAny>,
}

/// Idle compile grammars (each with its own code builder) beyond which the
/// pool does not keep the ones that concurrent or nested fallbacks made.
const MAX_IDLE_GRAMMARS: usize = 16;

/// What the fast path produced for one call.
enum Attempt {
    /// pybars' `FunctionContainer` for the code.
    Container(Py<PyAny>),
    /// pybars' `PybarsError`, to raise.
    Raise(PyErr),
    /// Run pybars instead.
    Fallback(Reason),
}

/// `Compiler._generate_code(self, source)`, as a callable that the Python
/// package binds as a method without adding a Python frame.
///
/// It runs the port with the GIL released and returns pybars' own
/// `FunctionContainer`, or raises `pybars.PybarsError` with pybars' message.
/// Where the port does not reproduce pybars (see `Reason`), it calls
/// pybars' original method from native code, so that it runs at the stack
/// depth it would have had without pybars3-rs and raises exactly what pybars
/// raises, `RecursionError` included.
#[pyclass(frozen, module = "pybars_rs._native")]
pub(crate) struct GenerateCode {
    pybars: Pybars,
    /// The verified pybars release, Unicode database, and default hooks, or
    /// `None` when the port is disabled.
    fast_path_config: Option<FastPath>,
    /// Compile grammars that no fallback is using. Each fallback takes one,
    /// so that fallbacks in several threads, or nested in one (a `str`
    /// subclass can compile a template while pybars parses it), never share
    /// a code builder. The mutex is never held while Python code runs.
    idle_grammars: Mutex<Vec<Py<PyAny>>>,
    sys: Py<PyAny>,
    recursion: Recursion,
    fast_path: AtomicU64,
    fallbacks: [AtomicU64; Reason::ALL.len()],
}

#[pymethods]
impl GenerateCode {
    /// Build from the installed pybars.
    ///
    /// * `original`: `pybars.Compiler._generate_code`, the fallback;
    /// * `compiler_class`: `pybars.Compiler`;
    /// * `compile_grammar_descriptor`: `pybars_rs.Compiler._compiler`
    ///   descriptor;
    /// * `function_container`: `pybars._compiler.FunctionContainer`;
    /// * `compiler_module`: `pybars._compiler`, for its `debug` flag;
    /// * `error_class`: `pybars.PybarsError`;
    /// * `fallback_grammar`: the `ContextVar` that `pybars_rs.Compiler._compiler`
    ///   reads;
    /// * `pybars_version`: the release whose compiler the installed pybars
    ///   has, such as `"0.9.7"`;
    /// * `unicode_version`: `unicodedata.unidata_version`.
    ///
    /// When `pybars_version` or `unicode_version` is `None`, the port is
    /// disabled, and every call runs `original`.
    #[new]
    #[pyo3(signature = (*, original, compiler_class, compile_grammar_descriptor, function_container, compiler_module, error_class, fallback_grammar, pybars_version, unicode_version))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        original: Py<PyAny>,
        compiler_class: Py<PyAny>,
        compile_grammar_descriptor: Py<PyAny>,
        function_container: Py<PyAny>,
        compiler_module: Py<PyAny>,
        error_class: Py<PyAny>,
        fallback_grammar: Py<PyAny>,
        pybars_version: Option<&str>,
        unicode_version: Option<&str>,
    ) -> PyResult<Self> {
        let fast_path_config = match (pybars_version, unicode_version) {
            (Some(pybars), Some(unicode)) => {
                let (pybars_version, unicode_version) = crate::parse_versions(pybars, unicode)?;
                let ws_name = match pybars_version {
                    PybarsVersion::V0_9_6 => "clean_whitespace",
                    PybarsVersion::V0_9_7 => "whitespace_control",
                    _ => unreachable!("verified pybars version"),
                };
                let cls = compiler_class.bind(py);
                let object_type = py.import("builtins")?.getattr("object")?;
                Some(FastPath {
                    pybars_version,
                    unicode_version,
                    object_getattribute: object_type.getattr("__getattribute__")?.unbind(),
                    whitespace_hook: cls.getattr(ws_name)?.unbind(),
                    handlebars: cls.getattr("_handlebars")?.unbind(),
                    extract_word: cls.getattr("_extract_word")?.unbind(),
                    original_compile_grammar: cls.getattr("_compiler")?.unbind(),
                    compile_grammar_descriptor,
                    compiler_class,
                    name_dict: PyString::intern(py, "__dict__").unbind(),
                    name_mro: PyString::intern(py, "__mro__").unbind(),
                    name_getattribute: PyString::intern(py, "__getattribute__").unbind(),
                    name_whitespace_hook: PyString::intern(py, ws_name).unbind(),
                    name_handlebars: PyString::intern(py, "_handlebars").unbind(),
                    name_extract_word: PyString::intern(py, "_extract_word").unbind(),
                    name_compiler: PyString::intern(py, "_compiler").unbind(),
                })
            }
            _ => None,
        };
        Ok(GenerateCode {
            pybars: Pybars {
                compiler_module,
                function_container,
                original,
                error_class,
                fallback_grammar,
            },
            fast_path_config,
            idle_grammars: Mutex::new(Vec::new()),
            sys: py.import("sys")?.into_any().unbind(),
            recursion: Recursion::of(py),
            fast_path: AtomicU64::new(0),
            fallbacks: Default::default(),
        })
    }

    /// `compiler._generate_code(source)`.
    fn __call__(
        &self,
        py: Python<'_>,
        compiler: &Bound<'_, PyAny>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match self.attempt(py, Some(compiler), source)? {
            Attempt::Container(container) => {
                self.fast_path.fetch_add(1, Ordering::Relaxed);
                Ok(container)
            }
            Attempt::Raise(error) => {
                self.fast_path.fetch_add(1, Ordering::Relaxed);
                Err(error)
            }
            Attempt::Fallback(reason) => {
                self.fallbacks[reason as usize].fetch_add(1, Ordering::Relaxed);
                self.fall_back(py, compiler, source)
            }
        }
    }

    /// How many calls the port answered (`fast_path`) and how many ran
    /// pybars, by reason.
    fn counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let counters = PyDict::new(py);
        counters.set_item("fast_path", self.fast_path.load(Ordering::Relaxed))?;
        for reason in Reason::ALL {
            counters.set_item(
                reason.name(),
                self.fallbacks[reason as usize].load(Ordering::Relaxed),
            )?;
        }
        Ok(counters)
    }

    /// The fast path alone, for tests and diagnostics: pybars'
    /// `FunctionContainer`, or the name of the reason pybars would run
    /// instead. Raises `PybarsError` as `__call__` does.
    fn try_fast_path(&self, py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match self.attempt(py, None, source)? {
            Attempt::Container(container) => Ok(container),
            Attempt::Raise(error) => Err(error),
            Attempt::Fallback(reason) => Ok(PyString::new(py, reason.name()).into_any().unbind()),
        }
    }

    /// The deepest nesting that the fast path compiles when called from the
    /// current stack depth.
    fn nesting_limit(&self, py: Python<'_>) -> PyResult<usize> {
        self.recursion.headroom(py, self.sys.bind(py))
    }
}

impl GenerateCode {
    fn attempt(
        &self,
        py: Python<'_>,
        compiler: Option<&Bound<'_, PyAny>>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<Attempt> {
        let Some(config) = &self.fast_path_config else {
            return Ok(Attempt::Fallback(Reason::Unavailable));
        };
        let Ok(source) = source.cast_exact::<PyString>() else {
            return Ok(Attempt::Fallback(Reason::NotStr));
        };
        if let Some(compiler) = compiler {
            if !config.is_default_compiler(py, compiler)? {
                return Ok(Attempt::Fallback(Reason::CustomCompiler));
            }
        }
        if self
            .pybars
            .compiler_module
            .bind(py)
            .getattr("debug")?
            .is_truthy()?
        {
            return Ok(Attempt::Fallback(Reason::Debug));
        }
        let Ok(text) = source.to_str() else {
            return Ok(Attempt::Fallback(Reason::Surrogates));
        };
        let options = Options::new(config.pybars_version, config.unicode_version)
            .with_int_max_str_digits(self.int_max_str_digits(py)?)
            .with_nesting_limit(self.recursion.headroom(py, self.sys.bind(py))?);
        // Copy the text out of the `str` before detaching from the
        // interpreter: the copy is cheap next to the compile.
        let text = text.to_owned();
        let Ok(result) =
            py.detach(move || catch_unwind(|| pybars3_compiler::precompile(&text, &options)))
        else {
            return Ok(Attempt::Fallback(Reason::Panic));
        };
        Ok(match result {
            // pybars' `FunctionContainer` adds the header of the pybars that
            // is installed, which is this release's.
            Ok(code) => match code.strip_prefix(config.pybars_version.header()) {
                Some(body) => {
                    Attempt::Container(self.pybars.function_container.call1(py, ("render", body))?)
                }
                None => Attempt::Fallback(Reason::Internal),
            },
            Err(Error::Template(error)) => Attempt::Raise(PyErr::from_value(
                self.pybars.error_class.bind(py).call1((error.message(),))?,
            )),
            Err(Error::Fallback(fallback)) => Attempt::Fallback(fallback.into()),
            // `Error` is non-exhaustive: a new kind of error runs pybars too.
            Err(_) => Attempt::Fallback(Reason::Internal),
        })
    }

    /// pybars' own `_generate_code`, with a compile grammar (and so a code
    /// builder) that no other call is using.
    ///
    /// A one-element list slot is placed in the `ContextVar` without running
    /// Python code, so pybars runs at the stack depth it would have had
    /// without pybars3-rs even if the pool is empty; if pybars reaches
    /// `self._compiler`, the descriptor populates the slot on demand.
    fn fall_back(
        &self,
        py: Python<'_>,
        compiler: &Bound<'_, PyAny>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.recursion.with_pybars_budget(py, || {
            let initial = match self.take_idle_grammar() {
                Some(grammar) => grammar,
                None => py.None(),
            };
            let slot = PyList::new(py, [initial])?;
            let fallback_grammar = self.pybars.fallback_grammar.bind(py);
            let token = fallback_grammar.call_method1("set", (&slot,))?;
            let result = self.pybars.original.call1(py, (compiler, source));
            let reset = fallback_grammar.call_method1("reset", (token,));
            if let Ok(grammar) = slot.get_item(0) {
                if !grammar.is_none() {
                    self.put_idle_grammar(grammar.unbind());
                }
            }
            reset?;
            result
        })
    }

    fn take_idle_grammar(&self) -> Option<Py<PyAny>> {
        self.idle_grammars
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
    }

    fn put_idle_grammar(&self, grammar: Py<PyAny>) {
        let mut idle = self
            .idle_grammars
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if idle.len() < MAX_IDLE_GRAMMARS {
            idle.push(grammar);
        }
    }

    /// `sys.get_int_max_str_digits()`, which a program can change at any
    /// time (0 when the interpreter has no limit).
    fn int_max_str_digits(&self, py: Python<'_>) -> PyResult<usize> {
        let sys = self.sys.bind(py);
        if !sys.hasattr("get_int_max_str_digits")? {
            return Ok(0);
        }
        sys.call_method0("get_int_max_str_digits")?.extract()
    }
}

/// Look up `name` in `mro` via each class's `__dict__`, without invoking
/// descriptors.
fn lookup_in_mro<'py>(
    mro: &Bound<'py, PyTuple>,
    dict_name: &Bound<'py, PyString>,
    name: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    for base in mro.iter() {
        let dict = base.getattr(dict_name)?;
        if dict.contains(name)? {
            return Ok(Some(dict.get_item(name)?));
        }
    }
    Ok(None)
}

impl FastPath {
    /// Whether `compiler` uses the unmodified hooks that pybars'
    /// `_generate_code` dispatches through.
    fn is_default_compiler(&self, py: Python<'_>, compiler: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !compiler.is_instance(self.compiler_class.bind(py))? {
            return Ok(false);
        }
        let cls = compiler.get_type();
        let mro = cls.getattr(self.name_mro.bind(py))?;
        let mro = mro.cast::<PyTuple>()?;
        let dict_name = self.name_dict.bind(py);
        // Check class-level hooks without invoking descriptors; verify
        // `__getattribute__` first so reading `compiler.__dict__` cannot run
        // user code.
        for (name, expected) in [
            (&self.name_getattribute, &self.object_getattribute),
            (&self.name_whitespace_hook, &self.whitespace_hook),
            (&self.name_handlebars, &self.handlebars),
            (&self.name_extract_word, &self.extract_word),
        ] {
            if !lookup_in_mro(mro, dict_name, name.bind(py))?
                .is_some_and(|found| found.is(expected.bind(py)))
            {
                return Ok(false);
            }
        }
        if !lookup_in_mro(mro, dict_name, self.name_compiler.bind(py))?.is_some_and(|attr| {
            attr.is(self.compile_grammar_descriptor.bind(py))
                || attr.is(self.original_compile_grammar.bind(py))
        }) {
            return Ok(false);
        }
        // Check the instance `__dict__` does not shadow any hook.
        let dict_obj = compiler.getattr(dict_name)?;
        let Ok(dict) = dict_obj.cast::<PyDict>() else {
            return Ok(false);
        };
        for name in [
            &self.name_whitespace_hook,
            &self.name_handlebars,
            &self.name_extract_word,
            &self.name_compiler,
        ] {
            if dict.contains(name.bind(py))? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
