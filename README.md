# Cuda Lexer

This is a cuda based parallel lexer of all languages. Some of lexer/regex related files about are extended from https://github.com/Snektron/pareas with significant changes. The cuda and lexer part are original to us.

## Compile && Run

Please login to nyu cims cuda2.cims.nyu.edu machine and run following command to compile in the root directory of this repo.

```bash
# compile the whole program using nvcc and g++
make clean && make
# run the program
./build/cuda_lexer
```

You will see something like 

```bash
Generating Merge Table...
```

which means that your .lex file has been used to generate merge table to be used for the lexical analyse.

After the generation completed, you will be asked to input a filename to be analysed. You can use `files/test3.json` as an example. The program will show you the file size and the execution time of both CUDA and CPU version of this lexical analysis and ask you input another filename again. Below is the output example.

```bash
Please input your filename: files/test3.json
--------------------------------------------------
Lexing files/test3.json (5240.49kb) using cuda and cpu
--------------------------------------------------
CUDA Running Time: 0.209865 s
lexeme		count
rbracket            	   72
lbracket            	   72
rbrace              	26257
true                	13441
number              	    1
comma               	85515
colon               	111700
string              	183750
whitespace          	249825
lbrace              	26257
CPU Running Time: 0.523647 s
lexeme		count
rbracket            	   72
lbracket            	   72
rbrace              	26257
true                	13441
number              	    1
comma               	85515
colon               	111700
string              	183750
whitespace          	249825
lbrace              	26257
```