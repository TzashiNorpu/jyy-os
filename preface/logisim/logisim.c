#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>

typedef bool wire; // Wires
typedef struct {
  bool value;
  wire *in, *out;
} reg; // Flip-flops


#define CLOCK       for (; ; END_CYCLE)
#define NAND(X, Y)  (!((X) && (Y)))
#define NOT(X)      (NAND(X, 1))
#define AND(X, Y)   (NOT(NAND(X, Y)))
#define OR(X, Y)    (NAND(NOT(X), NOT(Y)))
#define XOR(X,Y)     (OR((AND(NOT(X),Y)),(AND(X,NOT(Y))))) // Y = (!A & B) | (A & !B)


#define END_CYCLE ({ end_cycle(); putchar('\n'); fflush(stdout); sleep(1); })
#define PRINT(X) printf(#X " = %d; ", X)


wire X, Y, S, T, X1, Y1, S1, T1, A, B, C, D, E, F, G;

reg b0 = {.in = &X1, .out = &X};
reg b1 = {.in = &Y1, .out = &Y};
reg b2 = {.in = &S1, .out = &S};
reg b3 = {.in = &T1, .out = &T};


void end_cycle() {
  PRINT(A); PRINT(B); PRINT(C); PRINT(D);
  PRINT(E); PRINT(F); PRINT(G);
}
// XYST【高 -> 低】
int main() {
  CLOCK {
    X1 = OR(AND(AND(S,T),Y),AND(X,AND(NOT(Y),AND(NOT(S),NOT(T))))); 
    Y1 = XOR(AND(S,T),Y);
    S1 = AND(XOR(S,T),NOT(X));
    T1 = NOT(T);
/*     PRINT(X1);
    PRINT(Y1);
    PRINT(S1);
    PRINT(T1);
    putchar('\n'); */

    A=OR(OR(OR(AND(NOT(Y),NOT(T)),S),AND(Y,T)),X);
		B=OR(OR(NOT(Y),AND(NOT(S),NOT(T))),AND(S,T));
		C=OR(OR(NOT(S),T),Y);
		D=OR(OR(OR(OR(AND(NOT(Y),NOT(T)),AND(NOT(Y),S)),AND(S,NOT(T))),AND(AND(Y,NOT(S)),T)),X);
		E=OR(AND(NOT(Y),NOT(T)),AND(S,NOT(T)));
		F=OR(OR(OR(AND(NOT(S),NOT(T)),AND(Y,NOT(S))),AND(Y,NOT(T))),X);
		G=OR(OR(OR(AND(NOT(Y),S),AND(Y,NOT(S))),AND(Y,NOT(T))),X);

    b0.value = *b0.in;  
    b1.value = *b1.in;
    b2.value = *b2.in;
    b3.value = *b3.in;

    *b0.out = b0.value; 
    *b1.out = b1.value;
    *b2.out = b2.value;
    *b3.out = b3.value;
  }
}