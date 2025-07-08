https://github.com/zorp-corp/jock-lang

# Technical Overview and Comprehensive Summary of the Jock Language Repository

## Introduction

**Jock** is a high-level, friendly, and practical programming language designed to compile directly to the Nock instruction set architecture (ISA). The Jock language ecosystem is tightly integrated with the Urbit and Nockchain projects, providing a modern syntax and programming experience while targeting a minimalist, formally defined computational substrate. The repository analyzed here, `zorp-corp/jock-lang`, contains the core language, compiler, test framework, and a suite of tutorial and demonstration materials.

## Architecture and Ecosystem

### Core Components

- **Jock Compiler (`jockc`)**: Written in Hoon, this compiler translates Jock source code into Nock, the low-level combinator calculus used by Urbit and related systems.
- **Jock Test Framework (`jockt`)**: A dedicated tool for running and verifying language unit tests, also implemented in Hoon.
- **NockApp Integration**: Jock is built to operate within the NockApp architecture, allowing compiled programs to run on any Nock VM (e.g., Sword, Vere).

### Directory Structure

- **`common/hoon/`**: Contains core Hoon libraries and Jock demo/test programs.
- **`crates/jockc` and `crates/jockt`**: Rust crates for the Jock compiler and tester, including Hoon source and Rust command-line interfaces.
- **`img/`**: Branding and documentation images.
- **`README.md`, `PHILOSOPHY.md`, `ROADMAP.md`**: Documentation outlining language goals, design philosophy, and future plans.

## Language Design and Philosophy

### Design Principles

- **Legibility and Practicality**: Jock emphasizes readable, modern syntax inspired by Swift, Ruby, Python, and Rust.
- **Direct Compilation to Nock**: No runtime or preprocessing dependencies are required; Jock compiles directly to Nock, facilitating formal reasoning and verifiability.
- **Type System**: Jock features a static type system with support for atoms, lists, sets, user-defined classes, and protocols (traits).
- **Interoperability**: Built-in support for importing and calling Hoon libraries, enabling seamless integration with the Urbit ecosystem.

### Inspirations

- **Syntax and Semantics**: The language borrows concepts from modern programming languages, offering familiar constructs such as `let`, `func`, `class`, `if/else`, and infix operators.
- **Functional and Object-Oriented Paradigms**: Jock supports both functional and object-oriented programming, including first-class functions, classes, methods, and protocols.

## Technical Features

### Syntax and Semantics

- **Arithmetic and Logical Operators**: Supports standard infix notation for arithmetic (`+`, `-`, `*`, `/`, `%`, `**`) and logical operations (`&&`, `||`, `!`, `^`).
- **Control Structures**: Includes `if/else`, `switch`, `match`, `loop`, and recursion constructs.
- **Data Structures**:
  - **Atoms**: Numbers, booleans, strings.
  - **Lists**: Homogeneous or heterogeneous, with indexing and slicing planned.
  - **Sets and Maps**: Native set and map types with type-safe operations.
  - **Classes and Objects**: User-defined types with encapsulated state and methods.
  - **Protocols (Traits)**: Compile-time interfaces for method and type validation.
- **Interoperability with Hoon**: Jock can import Hoon libraries and call Hoon functions directly.

### Compilation Pipeline

- **Tokenization**: Converts source code into a sequence of tokens (keywords, literals, names, types, punctuators).
- **Parsing**: Builds an abstract syntax tree (AST) representing the program structure.
- **Type Checking**: Infers and validates types throughout the AST.
- **Nock Code Generation**: Produces Nock code, which can be executed on any compatible VM.
- **Testing and Debugging**: Integrated test framework (`jockt`) supports unit tests, debugging, and detailed output at each compilation stage.

### Example: Arithmetic Operations

```jock
[
    1 + 2
    4 - 3
    6 * 5
    8 / 7
    9 ** 4
    11 % 12
]
```
This code demonstrates Jock's support for infix arithmetic and list literals.

### Example: Class Definition and Usage

```jock
compose
  class Point(x:@ y:@) {
    add(p:(x:@ y:@)) -> Point {
      (x + p.x
       y + p.y)
    }
    sub(p:(x:@ y:@)) -> Point {
      (x - p.x
       y - p.y)
    }
  };

let pt = Point(100 100);
let st = pt.add(50 80);
(st Point(20 30))
```
This code illustrates defining a class with methods and instantiating objects.

## Tooling and Developer Workflow

### Building and Running Jock Code

1. **Compiler Setup**: Build the compiler with `make jockc`.
2. **Running Programs**: Execute Jock files using `./jockc path/to/file --import-dir path/to/libs`.
3. **Testing**: Use `jockt` to run tests, check parsing, AST generation, compilation, and execution.
4. **Library Imports**: Libraries can be imported from specified directories, supporting both Hoon and Jock code.

### Output Stages

- **%parse**: Token stream.
- **%jeam**: Jock AST.
- **%mint**: Compiled Nock code.
- **%jype**: Inferred or declared type.
- **%nock**: Evaluated result as a Nock atom.

## Roadmap and Planned Features

- **Caching**: Avoid redundant library builds to improve compilation speed.
- **Debugging**: Per-line traceback and improved error reporting.
- **Floating-Point Support**: Native fractional number types.
- **Enhanced Data Structures**: Full support for maps, strings, and advanced list operations.
- **Operator Overloading**: Via protocols/traits.
- **REPL/CLI**: Interactive shell for rapid development.
- **Generalized Compiler Namespace**: Improved handling of libraries and arguments.
- **Print and Side Effects**: Improved output capabilities.
- **Path and Wire Types**: For advanced program composition and messaging.

## Integration with Nock and Urbit

- **Nock ISA**: Jock compiles to Nock, a minimalist combinator calculus with twelve opcodes, enabling formal verification and efficient execution.
- **NockApp and Nockchain**: Jock programs can run within the NockApp framework, leveraging the Sword and Vere VMs.
- **Urbit Compatibility**: Direct interoperability with Urbit's Hoon language and ecosystem.

## Testing and Quality Assurance

- **Extensive Test Suite**: Located in `crates/jockt/hoon/lib/tests/`, covering all language features, edge cases, and integration points.
- **Automated Test Runner**: `jockt` can execute individual tests, all tests, or specific compilation stages for thorough validation.
- **Sample Programs**: Includes demos for arithmetic, Fibonacci sequence, class usage, and more.

## Licensing and Community

- **MIT License**: Open source, permissive licensing.
- **Active Development**: Developer preview released in October 2024, with an alpha planned for June 2025.
- **Community Resources**: Documentation, tutorials, and community engagement via the Jock language site and social media.

## Conclusion

Jock represents a modern, high-level language for the Nock ecosystem, combining expressive syntax, a robust type system, and seamless integration with Urbit and Hoon. Its architecture enables both practical programming and formal reasoning, making it a compelling choice for developers targeting Nock-based platforms. The repository provides a comprehensive foundation for language development, testing, and future expansion[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/343060/6c2825a0-7e1f-40b8-bfdb-985b88a60757/paste.txt
[2] https://github.com/zorp-corp/jock-lang
[3] https://www.reddit.com/r/Julia/comments/odr8mh/the_julia_programming_language_repository_summary/
[4] https://idratherbewriting.com/2014/12/22/most-common-programming-languages-tech-writers-in-my-survey-know/
[5] https://www.youtube.com/watch?v=CpJf_6nWqLk
[6] https://drops.dagstuhl.de/storage/00lipics/lipics-vol136-snapl2019/LIPIcs.SNAPL.2019.7/LIPIcs.SNAPL.2019.7.pdf
[7] https://www.digitalocean.com/community/tutorials/nodejs-testing-http-requests-with-nock
[8] http://abacus.bates.edu/acad/depts/biobook/Hockett.htm
[9] https://www.infoq.com/presentations/j-language/
[10] https://www.nockchain.org/jock-and-awe/
[11] https://en.wikipedia.org/wiki/Hockett's_design_features
[12] https://www.linkedin.com/pulse/tech-humor-101-guide-leaders-gilles-crofils-mxmye
[13] https://developers.urbit.org/reference/nock/explanation
[14] https://pmc.ncbi.nlm.nih.gov/articles/PMC4544681/
[15] https://en.wikipedia.org/wiki/APL_(programming_language)
[16] https://esolangs.org/wiki/Language_list
[17] https://en.wikipedia.org/wiki/Jock_(stereotype)
[18] https://www.matthewjockers.net
[19] https://developers.urbit.org/reference/glossary/nock
[20] https://www.studysmarter.co.uk/explanations/english/language-and-social-groups/eckert-jocks-and-burnouts/
[21] https://wejn.org/2021/02/urbit-good-bad-insane/

## Jock Language: Technical Deep Dive with Code Snippets

### Overview

Jock is a high-level, statically typed programming language that compiles directly to the Nock instruction set. It is designed for clarity, practicality, and seamless integration with the Urbit and Nockchain ecosystems. Below is a detailed technical exploration, with code snippets illustrating key features and mechanisms.

### 1. Language Syntax and Semantics

#### Arithmetic and Logical Operators

Jock supports standard infix notation for arithmetic and logical operations. Operators resolve to Hoon functions or class methods.

```jock
// Arithmetic operations in a list
[
    1 + 2
    4 - 3
    6 * 5
    8 / 7
    9 ** 4
    11 % 12
]
```

#### Control Structures

Jock provides familiar control flow constructs:

```jock
let a: @ = 3;

if a == 3 {
  42
} else {
  17
}
```

```jock
let a: @ = 3;

if a == 3 {
  42
} else if a == 5 {
  17
} else {
  15
}
```

#### Data Structures

- **Atoms**: Numbers, booleans, strings.
- **Lists**: Homogeneous or heterogeneous.

```jock
let a = [1 2 3 4 5];
let b:List(@ @) = [(10 20) (30 40) (50 60)];
```

- **Sets**:

```jock
let a:Set(@) = {1};
let b:Set(@) = {1 2};
let d:Set((@ @)) = {(1 2) (3 4) (1 2)};
```

#### Classes and Methods

Jock supports object-oriented programming with class definitions and methods:

```jock
compose
  class Point(x:@ y:@) {
    add(p:(x:@ y:@)) -> Point {
      (x + p.x
       y + p.y)
    }
    sub(p:(x:@ y:@)) -> Point {
      (x - p.x
       y - p.y)
    }
  };

let pt = Point(100 100);
let st = pt.add(50 80);
(st Point(20 30))
```

#### Functions and Lambdas

```jock
func fib(n:@) -> @ {
  if n == 0 {
    1
  } else if n == 1 {
    1
  } else {
    $(n - 1) + $(n - 2)
  }
};

fib(10)
```

```jock
lambda (b:@) -> @ {
  +(b)
}(41)
```

### 2. Compilation Pipeline

Jock's compilation process involves several distinct stages:

#### Tokenization

Converts source code into a stream of tokens:

```hoon
++  tokenize
  |=  txt=@
  ^-  tokens
  (rash txt parse-tokens)
```

#### Parsing

Builds an abstract syntax tree (AST):

```hoon
++  jeam
  |=  txt=@
  ^-  jock
  =+  [jok tokens]=(~(match-jock +>+>+ libs) (rash txt parse-tokens))
  ?.  ?=(~ tokens)
    ~|  'jeam: must parse to a single jock'
    ~|  remaining+tokens
    !!
  jok
```

#### Type Checking

Types are inferred and validated through the AST:

```hoon
++  jypist
  |=  txt=@
  ^-  jype
  =/  jok  (jeam (cat 3 'import hoon;\0a' txt))
  =+  [nok jyp]=(~(mint cj:~(. +> libs) [%atom %string %.n]^%$) jok)
  jyp
```

#### Nock Code Generation

Generates executable Nock code:

```hoon
++  mint
  |=  txt=@
  ^-  *
  =/  jok  (jeam (cat 3 'import hoon;\0a' txt))
  =+  [nok jyp]=(~(mint cj:~(. +> libs) [%atom %string %.n]^%$) jok)
  nok
```

### 3. Example: Class and Method Usage

```jock
compose
  class Point(x:@ y:@) {
    add(p:(x:@ y:@)) -> Point {
      (x + p.x
       y + p.y)
    }
  }
;

let point_1 = Point(14 104);
point_1 = point_1.add(28 38);
(point_1.x() point_1.y())
```

Equivalent Hoon (Urbit's language) for reference:

```hoon
=>
  ^=  door
  |_  [x=@ y=@]
  ++  add
    |=  p=[x=@ y=@]
    [(add:mini x x.p) (add:mini y y.p)]
  --
=/  point-1
  ~(. door [14 104])
=.  point-1  ~(. door (add:point-1 [28 38]))
[+12 +13]:point-1
```

### 4. Interoperability with Hoon

Jock can import and call Hoon libraries directly:

```jock
import hoon;

let a:@ = 5;
let b:@ = 37;

(
  hoon.dec(43)
  hoon.add(5 37)
  hoon.add(a b)
  hoon.sub(47 a)
  hoon.lent([1 2 3 4 5 6 7 8 9 10])
)
```

### 5. Testing and Debugging

Jock includes an integrated test framework (`jockt`), which can execute and validate code at every compilation stage.

```sh
make jockt
./jockt exec 5 --import-dir ./common/hoon/jib
./jockt test-all --import-dir ./common/hoon/jib
```

### 6. Compilation Output Stages

When running a Jock program, the following outputs are produced:

- `%parse`: Token stream
- `%jeam`: Jock AST
- `%mint`: Compiled Nock code
- `%jype`: Inferred or declared type
- `%nock`: Evaluated result as a Nock atom

### 7. Advanced Features and Planned Roadmap

- **Operator Overloading** (planned):

```jock
protocol Arithmetic {
    add(# #) -> #;
    sub(# #) -> #;
    bitwidth(#) -> ##;
};
```

- **Enhanced Data Structures**: Maps, strings, advanced list operations
- **REPL/CLI**: Interactive shell for rapid development
- **Debugging**: Per-line traceback and improved error reporting

### 8. Build and Run Workflow

#### Building the Compiler

```sh
make jockc
cp ./target/release/jockc .
```

#### Running a Program

```sh
./jockc ./common/hoon/try/hello-world --import-dir ./common/hoon/jib
```

#### Adding a Demo

1. Save your `.jock` file in `/common/hoon/try`
2. Run it by name:

```sh
./jockc ./common/hoon/try/my-demo --import-dir ./common/hoon/jib
```

### 9. Example: Fibonacci in Jock

```jock
func fib(n:@) -> @ {
  if n == 0 {
    1
  } else if n == 1 {
    1
  } else {
    $(n - 1) + $(n - 2)
  }
};

(
  fib(0)
  fib(1)
  fib(2)
  fib(3)
  fib(4)
  fib(5)
  fib(6)
  fib(7)
  fib(8)
  fib(9)
  fib(10)
)
```

### 10. Licensing and Community

- **MIT License**: Open source, permissive
- **Active Development**: Developer preview released October 2024, alpha planned for June 2025

Jock offers a modern, expressive, and technically robust language for Nock-based systems, with a focus on formal reasoning, practical syntax, and deep integration with Urbit and Hoon. The repository provides extensive code, tests, and documentation for both language users and contributors[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/343060/6c2825a0-7e1f-40b8-bfdb-985b88a60757/paste.txt