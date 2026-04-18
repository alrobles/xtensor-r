# xtensor-r: R Bindings Implementation Analysis

## 1. Repository Structure

`xtensor-r` is a **header-only C++ binding layer** over [xtensor](https://github.com/xtensor-stack/xtensor). It is not an R package itself — the R package boilerplate lives in a separate repository ([Xtensor.R](https://github.com/xtensor-stack/Xtensor.R)), as noted in `README.md:68-72`.

### Source layout

| Path | Purpose |
|------|---------|
| `include/xtensor-r/*.hpp` | All binding headers (7 files) |
| `CMakeLists.txt` | Build system; declares an `INTERFACE` library linked to `xtensor` (lines 56-62) |
| `cmake/FindR.cmake` | Locates the R installation (R command, headers, libs) |
| `test/` | GoogleTest + RInside C++ tests and Rcpp-based R tests |
| `docs/` | Sphinx + Doxygen documentation |
| `environment-dev.yml` | Conda/mamba dev environment pinning `xtensor`, `r-base`, `r-rcpp`, etc. |

The library target is `INTERFACE` (header-only) and links to `xtensor`:

```cmake
# CMakeLists.txt:56-62
add_library(xtensor-r INTERFACE)
target_include_directories(xtensor-r INTERFACE ...)
target_compile_features(xtensor-r INTERFACE cxx_std_20)
target_link_libraries(xtensor-r INTERFACE xtensor)
```

### Header inventory

| Header | Role |
|--------|------|
| `rcontainer.hpp` | Shared CRTP base for R-backed containers |
| `rarray.hpp` | Dynamic-rank container (analogous to `xarray`) |
| `rtensor.hpp` | Fixed-rank container (analogous to `xtensor`) |
| `rcpp_extensions.hpp` | Rcpp trait specializations (`Exporter`, `wrap`) |
| `roptional.hpp` | NA-aware optional container wrappers |
| `rvectorize.hpp` | Numpy-style universal function adapter for R |
| `xtensor_r_config.hpp` | Version macros and config flags |

---

## 2. Integration with xtensor Core

The binding strategy is: **wrap R `SEXP` memory as xtensor containers** while preserving full xtensor expression semantics.

### 2.1 Container hierarchy

```
xcontainer<D>  (xtensor core)
      │
      ├── rcontainer<D, SP>  (rcontainer.hpp:123-125)
      │        │
      │        ├── rarray<T>      (rarray.hpp:71-73)
      │        │
      │        └── rtensor<T,N>   (rtensor.hpp:69-71)
      │
Rcpp::PreserveStorage<D>  (Rcpp storage policy)
```

`rcontainer` inherits from **both** `xcontainer<D>` (xtensor's container base) and `Rcpp::PreserveStorage<D>` (Rcpp's GC-safe storage policy):

```cpp
// rcontainer.hpp:123-124
template <class D, template <class> class SP = Rcpp::PreserveStorage>
class rcontainer : public xcontainer<D>, public SP<D>
```

### 2.2 Zero-copy memory adaptation

Both containers define `storage_type` as `xbuffer_adaptor<...*>`, creating a **view over existing R memory** rather than copying:

```cpp
// rarray.hpp:36-37
using storage_type = xbuffer_adaptor<typename underlying_type::type*>;

// rtensor.hpp:40-41
using storage_type = xbuffer_adaptor<typename underlying_type::type*>;
```

The `update(SEXP)` method maps R's raw memory pointer into this adaptor and computes xtensor strides:

```cpp
// rarray.hpp:365-372
template <class T>
inline void rarray<T>::update(SEXP new_sexp) noexcept
{
    this->m_shape = detail::r_shape_to_buffer_adaptor(new_sexp);
    resize_container(m_strides, m_shape.size());
    resize_container(m_backstrides, m_shape.size());
    std::size_t sz = xt::compute_strides<layout_type::column_major>(
        m_shape, layout_type::column_major, m_strides, m_backstrides);
    this->m_storage = storage_type(
        reinterpret_cast<value_type*>(
            Rcpp::internal::r_vector_start<SXP>(new_sexp)), sz);
}
```

### 2.3 Column-major layout

R stores arrays in column-major order. All binding containers enforce this:

```cpp
// rcontainer.hpp:166
static constexpr layout_type static_layout = layout_type::column_major;

// rcontainer.hpp:232-235
template <class D, template <class> class SP>
inline layout_type rcontainer<D, SP>::layout() const
{
    return layout_type::column_major;
}
```

### 2.4 Shape from R's `dim` attribute

Shape is extracted from R's `R_DimSymbol` attribute, with a fallback to vector length:

```cpp
// rcontainer.hpp:77-84
inline xbuffer_adaptor<int*> r_shape_to_buffer_adaptor(SEXP exp)
{
    SEXP dim = Rf_getAttrib(exp, R_DimSymbol);
    SEXP shape_sexp = Rf_isNull(dim)
        ? SEXP(Rcpp::IntegerVector::create(Rf_length(exp)))
        : dim;
    std::size_t n = (std::size_t)Rf_xlength(shape_sexp);
    return xbuffer_adaptor<int*>(
        Rcpp::internal::r_vector_start<INTSXP>(shape_sexp), n);
}
```

`rarray::reshape()` writes back to `R_DimSymbol` so reshapes are reflected on the R side:

```cpp
// rcontainer.hpp:224-228
auto tmp_shape = Rcpp::IntegerVector(std::begin(shape), std::end(shape));
Rf_setAttrib(rstorage::get__(), R_DimSymbol, SEXP(tmp_shape));
this->derived_cast().update_shape_and_strides();
```

---

## 3. Rcpp Interop Layer

### 3.1 `Rcpp::traits::Exporter` specializations

`rcpp_extensions.hpp` prevents copies in Rcpp's generic `as<T>` by specializing `Exporter`:

```cpp
// rcpp_extensions.hpp:34-52
template <class T>
class Exporter<xt::rarray<T>>
{
public:
    Exporter(SEXP x) : m_sexp(x) {}
    inline xt::rarray<T> get() { return xt::rarray<T>(m_sexp); }
private:
    SEXP m_sexp;
};

// rcpp_extensions.hpp:54-72
template <class T, std::size_t N>
class Exporter<xt::rtensor<T, N>>
{
public:
    Exporter(SEXP x) : m_sexp(x) {}
    inline xt::rtensor<T, N> get() { return xt::rtensor<T, N>(m_sexp); }
private:
    SEXP m_sexp;
};
```

A third specialization handles optional containers (`rcpp_extensions.hpp:74-93`).

### 3.2 `Rcpp::wrap` overloads

Return direction (C++ → R) uses `wrap` specializations that cast the container back to `SEXP`:

```cpp
// rarray.hpp:384-388
namespace Rcpp {
    template <typename T>
    inline SEXP wrap(const xt::rarray<T>& arr) { return SEXP(arr); }
}

// rtensor.hpp:367-371
namespace Rcpp {
    template <typename T, std::size_t N>
    inline SEXP wrap(const xt::rtensor<T, N>& arr) { return SEXP(arr); }
}
```

### 3.3 R memory allocation

Constructors allocate R arrays via `Rf_allocArray` / `Rf_allocVector` and store them through the Rcpp storage policy:

```cpp
// rarray.hpp:187-193
template <class T>
template <class S>
inline void rarray<T>::init_from_shape(const S& shape)
{
    if (shape.size() == 0)
        base_type::rstorage::set__(Rf_allocVector(SXP, 1));
    else {
        Rcpp::IntegerVector tmp_shape(shape.begin(), shape.end());
        base_type::rstorage::set__(Rf_allocArray(SXP, SEXP(tmp_shape)));
    }
}
```

---

## 4. Type System and R-specific Behavior

### 4.1 Allowed types

R supports only a subset of C++ types. A `static_assert` enforces this:

```cpp
// rcontainer.hpp:144-151
static_assert(std::disjunction<
    std::is_same<r_type, int32_t>,
    std::is_same<r_type, double>,
    std::is_same<r_type, Rbyte>,
    std::is_same<r_type, rlogical>,
    std::is_same<r_type, std::complex<double>>
>::value == true,
"R containers can only be of type rlogical, int, double, std::complex<double>.");
```

### 4.2 R logical handling

R stores logicals as `int32`. A fake type `rlogical` and a trait specialization handle this:

```cpp
// rcontainer.hpp:29
struct rlogical {};

// rcontainer.hpp:38-43
template<>
struct r_sexptype_traits<rlogical> {
    static constexpr int rtype = LGLSXP;
};

// rcontainer.hpp:62-72
template <>
struct get_underlying_value_type_r<rlogical> {
    using type = int;  // stored as int internally
};
```

### 4.3 Coercion warnings

Type coercion (e.g., passing a `REALSXP` where `INTSXP` is expected) issues warnings when `XTENSOR_WARN_ON_COERCE` is enabled:

```cpp
// rcontainer.hpp:98-109
template <int SXP>
inline void check_coercion(SEXP exp)
{
#if XTENSOR_WARN_ON_COERCE
    if (TYPEOF(exp) != SXP) {
        Rcpp::warning("Coerced object from '%s' to '%s'...",
            Rf_type2char(TYPEOF(exp)), Rf_type2char(SXP));
    }
#endif
}
```

This is enabled by default (`xtensor_r_config.hpp:17-19`).

### 4.4 Type mapping summary

| R type | C++ type | SEXPTYPE |
|--------|----------|----------|
| Integer | `int` / `int32_t` | `INTSXP` |
| Real | `double` | `REALSXP` |
| Complex | `std::complex<double>` | `CPLXSXP` |
| Raw | `Rbyte` / `uint8_t` | `RAWSXP` |
| Logical | `rlogical` (→ `int`) | `LGLSXP` |

---

## 5. Optional / NA Bindings

R uses special sentinel values (e.g., `NA_REAL`, `NA_INTEGER`) to represent missing data. `xtensor-r` bridges this to xtensor's optional system.

### 5.1 Container aliases

```cpp
// roptional.hpp:35-38
template <class T>
using rarray_optional = rcontainer_optional<rarray<T>>;

template <class T, std::size_t N>
using rtensor_optional = rcontainer_optional<rtensor<T, N>>;
```

### 5.2 NA proxy

`rna_proxy` provides boolean semantics for "has value" by checking R's `is_na`:

```cpp
// roptional.hpp:47-75
template <class XT>
struct rna_proxy {
    inline operator bool() const {
        return !Rcpp::traits::is_na<rtype>(m_val);
    }
    inline rna_proxy& operator=(bool val) {
        if (val == false) m_val = Rcpp::traits::get_na<rtype>();
        else if (m_val == Rcpp::traits::get_na<rtype>()) m_val = 0;
        return *this;
    }
};
```

### 5.3 Functor-based flag expression

A `rna_proxy_functor` wraps each element to produce the "has_value" view, composed into an `xfunctor_adaptor`:

```cpp
// roptional.hpp:134-138
using raw_flag_expression = xfunctor_adaptor<rna_proxy_functor<value_type>, RC&>;
using flag_storage_type = xfunctor_adaptor<rna_proxy_functor<value_type>, RC&>;
using storage_type = xoptional_assembly_storage<value_storage_type&, flag_storage_type&>;
```

### 5.4 Usage example (from tests)

```cpp
// test/test_roptional.cpp:22-33
rarray<double> t {{{ 0, 1, 2}, { 3, mi, 5}, ...}};
SEXP exp = SEXP(t);
rarray_optional<double> o(exp);
EXPECT_TRUE(o(0, 0, 0).has_value());
EXPECT_FALSE(o(0, 1, 1).has_value());  // mi = NA_REAL
```

---

## 6. Vectorized Function Adapter

`rvectorize` creates numpy-style universal functions that operate on `rarray`:

```cpp
// rvectorize.hpp:22-37
template <class Func, class R, class... Args>
struct rvectorizer {
    xvectorizer<Func, R> m_vectorizer;

    inline rarray<R> operator()(const rarray<Args>&... args) const {
        rarray<R> res = m_vectorizer(args...);
        return res;
    }
};

// rvectorize.hpp:43-46
template <class R, class... Args>
inline rvectorizer<R (*)(Args...), R, Args...> rvectorize(R (*f)(Args...)) {
    return rvectorizer<R (*)(Args...), R, Args...>(f);
}
```

Supports function pointers, lambdas, and callable objects. Tested at `test/test_rvectorize.cpp:30-59`.

---

## 7. Build and Test Infrastructure

### 7.1 CMake build

- Requires `xtensor ≥ 0.27.1` and `xtl ≥ 0.8.1` (`CMakeLists.txt:19-20`).
- Custom `FindR.cmake` discovers R, headers, and libraries (`cmake/FindR.cmake:40-94`).
- Rcpp and RInside are located by invoking R at configure time (`CMakeLists.txt:77-114`).

### 7.2 C++ test harness

Tests embed an R interpreter via RInside before running GoogleTest:

```cpp
// test/main.cpp:20-25
RInside R(argc, argv);
::testing::InitGoogleTest(&argc, argv);
int ret = RUN_ALL_TESTS();
```

Test suite covers: `test_rarray.cpp`, `test_rtensor.cpp`, `test_rvectorize.cpp`, `test_roptional.cpp`, `test_rreducer.cpp`, `test_sfinae.cpp`.

### 7.3 R integration tests

`test/rcpp_tests.cpp` uses `// [[Rcpp::export]]` annotations to expose C++ functions:

```cpp
// test/rcpp_tests.cpp:19-25
// [[Rcpp::export]]
int modify_cpp(xt::rarray<double>& x) {
    x(0, 0) = -1000;
    x(9, 2) = 1000;
    return 1;
}
```

These are loaded in R via `Rcpp::sourceCpp` and validated with `testthat`:

```r
# test/unittest.R:3-14
Rcpp::sourceCpp("rcpp_tests.cpp", cacheDir="__cache__")
test_that("cpp_inplace", {
    expect_equal(modify_cpp(x), 1)
    expect_equal(x[1, 1], -1000)  # mutation visible in R
})
```

### 7.4 CI pipelines

| Platform | Compilers | File |
|----------|-----------|------|
| Linux | GCC 11-14, Clang 19-20 | `.github/workflows/linux.yml` |
| macOS | Default (14, 15) | `.github/workflows/osx.yml` |
| Windows | MinGW64 | `.github/workflows/windows.yml` |

All use `micromamba` with `environment-dev.yml` for reproducible deps.

### 7.5 Dev environment

```yaml
# environment-dev.yml
dependencies:
  - cmake
  - xtensor=0.27.1
  - r-base
  - r-rcpp
  - r-rinside
  - r-devtools
  - r-testthat
```

---

## 8. Key Design Patterns

### 8.1 CRTP (Curiously Recurring Template Pattern)

All containers use CRTP through xtensor's `xcontainer<D>` base. `rcontainer` passes the derived type (`rarray<T>` or `rtensor<T,N>`) as template argument, enabling static dispatch of `shape_impl()`, `strides_impl()`, `storage_impl()`, and `update()`.

### 8.2 Trait-based inner types

Each container specializes `xcontainer_inner_types<D>` to wire up storage, shape, and stride types:

```cpp
// rarray.hpp:32-51
template <class T>
struct xcontainer_inner_types<rarray<T>> {
    using storage_type = xbuffer_adaptor<...>;
    using shape_type = xt::dynamic_shape<...>;     // dynamic rank
    ...
};

// rtensor.hpp:36-52
template <class T, std::size_t N>
struct xcontainer_inner_types<rtensor<T, N>> {
    using storage_type = xbuffer_adaptor<...>;
    using shape_type = std::array<int, N>;          // fixed rank
    ...
};
```

### 8.3 Storage policy delegation

`Rcpp::PreserveStorage` manages R's GC protection (`PROTECT`/`UNPROTECT`) automatically. The `set__()` method triggers `update()` on the derived class, syncing xtensor state with the new SEXP.

### 8.4 Expression compatibility

Because `rarray` and `rtensor` inherit from `xcontainer_semantic`, they support xtensor expressions:

```cpp
// rarray.hpp:312-315
template <class T>
template <class E>
inline rarray<T>::rarray(const xexpression<E>& e) {
    semantic_base::assign(e);
}
```

This means `xt::sin(rarray)`, `rarray + xtensor`, broadcasting, etc. all work.

---

## 9. Summary

The `xtensor-r` repository implements R bindings as a **thin, header-only adaptation layer** that:

1. **Reuses xtensor's container/expression machinery** (`xcontainer`, `xsemantic`, `xvectorize`) via CRTP and trait specializations.
2. **Maps R memory/shape** (`SEXP`, `R_DimSymbol`) into xtensor's `xbuffer_adaptor` storage and stride system — enabling zero-copy, in-place operations.
3. **Uses targeted Rcpp trait specializations** (`Exporter`/`wrap`) for ergonomic and copy-free Rcpp interoperability.
4. **Adds R-specific semantics**: column-major layout enforcement, strict type mapping (`static_assert`), coercion warnings, logical-as-int handling, and NA-aware optional containers.
5. **Provides a vectorization adapter** (`rvectorize`) for creating universal functions from scalar C++ functions.

This design minimizes duplicated array logic and keeps most computation in the xtensor core while making R arrays first-class xtensor expressions.
