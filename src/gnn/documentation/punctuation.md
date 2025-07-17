# GNN Punctuation Specification
**Version: 1.0**

## Symbols

| Symbol | Meaning | ExampleUse | MeaningOfExample |
|--------|---------|------------|------------------|
| ^ | A caret means superscript. | X^Y | X with a superscript Y |
| , | A comma is used to separate items in a list. | X,Y | List with X and Y as elements |
| ## | A double hashtag signals a new section in the Markdown file. | ## Section123 | Has "Section123" as a section name |
| # | A hashtag signals the title header in the Markdown file. | # Title123 | Has "Title123" as model title |
| ### | A triple hashtag is a comment line in the Markdown file. | ### Comment123 | Has "Comment123" as a comment |
| {} | Curly brackets are specification of exact values for a variable. | X{1} | X equals 1 exactly |
| - | Hyphen is an undirected causal edge between two variables. | X-Y | Undirected relation between X and Y |
| () | Parentheses are used to group expressions. | X^(Y_2) | X with a superscript that is Y with a subscript 2 |
| [] | Rectangular brackets define the dimensionality, or state space, of a variable. | X[2,3] | X is a matrix with dimensions (2,3) |
| = | The equals sign declares equality or assignment. | X=5 | Sets the variable X to value of 5 |
| > | The greater than symbol represents a directed causal edge between two variables. | X>Y | Causal influence from X to Y |
| _ | Underscore means subscript. | X_2 | X with a subscript 2 |
| + | Plus sign for addition or other operations. | X+Y | Sum of X and Y |
| * | Asterisk for multiplication or other operations. | X*Y | Product of X and Y |
| / | Forward slash for division or other operations. | X/Y | X divided by Y |
| \| | Vertical bar for conditional probability or alternatives. | P(X\|Y) | Probability of X given Y |