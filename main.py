from logic import *

p = Symbol("P")
q = Symbol("Q")

axiom = And(
    Or(And(p, q), Not(q))
)

theorem = q

symbol_name_list = ", ".join(name.removeprefix("'").removesuffix("'") for name in str(axiom.symbols())[1:-1].split(", "))

print(
"Given symbols,\n\t" + 
    symbol_name_list + 
"\ndoes axiom\n\t" + 
    axiom.formula(), 
"\nprove theorem\n\t" + 
    theorem.formula() + 
"\n" + str(model_check(axiom, theorem)))

print("Axiom in CNF:\n\t" + convert_cnf(axiom).formula())
