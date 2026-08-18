# Klong: A Simple Array Language

## Overview

Klong is an array language inspired by K (which is in turn inspired by APL), but with unambiguous syntax. Unlike K, where semantic information is sometimes required to understand syntax, Klong's syntax alone is sufficient to understand program meaning.

**Key Principle**: Klong is a mathematical notation rather than a traditional programming language. It excels at manipulating lists (vectors) and multi-dimensional arrays through a rich set of operators and functions.

## Core Characteristics

### Unambiguous Syntax
- **K Example**: `f/x` could mean fold, fixpoint, while, or iterate depending on types
- **Klong Solution**: Uses distinct operators:
  - `f/x` (Over/Fold)
  - `f:~x` (Converge/Fixpoint)
  - `x f:~y` (While)
  - `x f:*y` (Iterate)

### Reduced Operator Overloading
- K's `x_y` means either "drop" or "cut" based on type
- Klong separates: `_` for drop, `:_` for cut

## Language Structure

### Data Types
- **Integers**: `123`, `-456`, `0b101010` (binary), `0o777` (octal), `0xcafe` (hex)
- **Real Numbers**: `123.45`, `1e6`, `6.62607004e-34`
- **Characters**: `0ca`, `0cA`, `0c*` (0c prefix)
- **Strings**: `"hello world"`, `"say ""hi""!"` (double quotes for quotes)
- **Symbols**: `:foo`, `:bar` (quoted symbols)
- **Lists**: `[1 2 3]`, `[1 [2 3] 4]` (nested allowed)
- **Arrays**: Multi-dimensional lists with symmetric shape
- **Dictionaries**: `:{[key1 value1] [key2 value2]}`

### Functions
Functions are first-class values defined with curly braces:

```klong
square::{x*x}           # Monad (1 argument)
add::{x+y}              # Dyad (2 arguments)  
combine::{x+y*z}        # Triad (3 arguments)
constant::{42}          # Nilad (0 arguments)
```

Function type determined by variables used:
- Contains `x`: monad
- Contains `y`: dyad  
- Contains `z`: triad
- Contains none: nilad

### Operators

#### Arithmetic
- `+` (plus), `-` (minus), `*` (times), `%` (divide)
- `^` (power), `!` (remainder), `:*` (integer divide)

#### Comparison  
- `=` (equal), `<` (less), `>` (more), `~` (match)
- `&` (min/and), `|` (max/or)

#### Array Operations
- `#` (size), `^` (shape), `:^` (reshape)
- `*` (first), `|` (reverse), `_` (drop), `#` (take)
- `@` (at/index), `:@` (index-in-depth)
- `,` (join), `:_` (cut), `:#` (split)

#### Advanced
- `?` (find), `!` (enumerate), `&` (expand/where)
- `=` (group), `<>` (grade up/down), `?` (range/unique)

### Adverbs
Adverbs modify verb behavior:

- `f'a` (Each): apply f to each element
- `f/a` (Over): fold f over list  
- `f\a` (Scan): like Over but collect intermediate results
- `f:~a` (Converge): find fixpoint of f
- `f:'a` (Each-Pair): apply f to consecutive pairs

### Example Programs

#### Prime Number Check (for x>2)
```klong
{&/x!:\2+!_x^1%2}
```

#### Flatten Any Nested Structure
```klong
,/:~
```

#### Square Root via Newton's Method  
```klong
{(x+2%x)%2}:~2
```

## Syntax Features

### Projection
Partial function application by omitting arguments:
```klong
f::{x-y}
f(5;)    # equivalent to {5-x}  
f(;5)    # equivalent to {x-5}
```

### Conditionals
```klong
:[predicate;consequent;alternative]
:[p1;c1:|p2;c2;alternative]  # else-if chains
```

### Comments
```klong
:"This is a comment"
```

## Installation & Usage

### Installation
1. Download from [t3x.org/klong](http://t3x.org/klong/)
2. Compile with `make && make test` (requires C99 compiler)
3. Copy `kg` binary to `/usr/local/bin`
4. Set `KLONGPATH` environment variable to `lib/` directory

### Running Programs
```bash
./kg -l filename     # Load and run file
./kg                 # Interactive mode
```

### File Extensions
- `.kg` files are Klong programs
- Load with `]lname` or `./kg -l name`

## Interactive Features

### REPL Commands
- `]q` - quit
- `]lfile` - load file  
- `]!command` - shell command
- `]h topic` - help on topic
- `it` - result of last computation

### Line Editing (if compiled in)
- Ctrl-A/E: beginning/end of line
- Ctrl-B/F: backward/forward character
- Ctrl-P/N: previous/next history
- Ctrl-H: backspace
- Ctrl-U: clear line

## Advanced Features

### Modules
Protect definitions from redefinition:
```klong
.module(:mymodule)
localvar::42
localfunc::{x*2}
.module(0)
```

### I/O Channels
- `.cin`, `.cout`, `.cerr` - standard streams
- `.ic(filename)` - input channel
- `.oc(filename)` - output channel  
- `.fc(channel)`, `.tc(channel)` - select channels

### System Integration
- `.sys("command")` - execute system command
- `.l("file")` - load Klong file
- `.E("code")` - evaluate string as Klong code

## Resources

### Documentation
- **Reference Manual**: Complete language specification
- **Quick Reference**: Syntax/semantics summary for K/APL users  
- **Introduction**: Beginner-friendly tutorial
- **Klong vs K**: Differences from K language

### Implementations
- **Original Klong**: Pure ANSI C implementation
- **KlongPy**: Vectorized Python implementation by Brian Gurraci
  - GitHub: [briangu/klongpy](https://github.com/briangu/klongpy)
  - Faster execution for performance-critical applications

### Philosophy
> "Klong is a mathematical notation rather than a programming language. If you try to use it like your favorite functional/procedural/OO programming language, you will only get frustrated."

Klong excels at:
- Array and list manipulation
- Mathematical computations
- Data transformation pipelines
- Concise algorithmic expression

## Integration with GNN

In the context of Generalized Notation Notation (GNN), Klong could serve as:

1. **Array Computation Backend**: Efficient manipulation of model matrices (A, B, C, D)
2. **State Space Operations**: Compact representation of state transitions
3. **Mathematical Notation**: Natural expression of Active Inference equations
4. **Data Pipeline Processing**: Transform and analyze model outputs

The unambiguous syntax and array-oriented design make Klong a potential candidate for implementing computational aspects of Active Inference models specified in GNN format.
