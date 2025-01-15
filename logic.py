import itertools

class Sentence():

    def evaluate(self, model):
        """Evaluates the logical sentence."""
        raise Exception("nothing to evaluate")

    def formula(self):
        """Returns string formula representing logical sentence."""
        return ""

    def symbols(self):
        """Returns a set of all symbols in the logical sentence."""
        return set()

    @classmethod
    def validate(cls, sentence):
        if not isinstance(sentence, Sentence):
            raise TypeError("must be a logical sentence")

    @classmethod
    def parenthesize(cls, s):
        """Parenthesizes an expression if not already parenthesized."""
        def balanced(s):
            """Checks if a string has balanced parentheses."""
            count = 0
            for c in s:
                if c == "(":
                    count += 1
                elif c == ")":
                    if count <= 0:
                        return False
                    count -= 1
            return count == 0
        if not len(s) or s.isalpha() or (
            s[0] == "(" and s[-1] == ")" and balanced(s[1:-1])
        ):
            return s
        else:
            return f"({s})"


class Symbol(Sentence):

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name == other.name

    def __hash__(self):
        return hash(("symbol", self.name))

    def __repr__(self):
        return self.name

    def evaluate(self, model):
        try:
            return bool(model[self.name])
        except KeyError:
            raise Exception(f"variable {self.name} not in model")

    def formula(self):
        return self.name
 
    def symbols(self):
        return {self.name}


class Not(Sentence):
    def __init__(self, operand):
        Sentence.validate(operand)
        self.operand = operand

    def __eq__(self, other):
        return isinstance(other, Not) and self.operand == other.operand

    def __hash__(self):
        return hash(("not", hash(self.operand)))

    def __repr__(self):
        return f"Not({self.operand})"

    def evaluate(self, model):
        return not self.operand.evaluate(model)

    def formula(self):
        return "¬" + Sentence.parenthesize(self.operand.formula())

    def symbols(self):
        return self.operand.symbols()


class And(Sentence):
    def __init__(self, *conjuncts):
        for conjunct in conjuncts:
            Sentence.validate(conjunct)
        self.conjuncts = list(conjuncts)

    def __eq__(self, other):
        return isinstance(other, And) and self.conjuncts == other.conjuncts

    def __hash__(self):
        return hash(
            ("and", tuple(hash(conjunct) for conjunct in self.conjuncts))
        )

    def __repr__(self):
        conjunctions = ", ".join(
            [str(conjunct) for conjunct in self.conjuncts]
        )
        return f"And({conjunctions})"

    def add(self, conjunct):
        Sentence.validate(conjunct)
        self.conjuncts.append(conjunct)

    def evaluate(self, model):
        return all(conjunct.evaluate(model) for conjunct in self.conjuncts)

    def formula(self):
        if len(self.conjuncts) == 1:
            return self.conjuncts[0].formula()
        return " ∧ ".join([Sentence.parenthesize(conjunct.formula())
                           for conjunct in self.conjuncts])

    def symbols(self):
        return set.union(*[conjunct.symbols() for conjunct in self.conjuncts])


class Or(Sentence):
    def __init__(self, *disjuncts):
        for disjunct in disjuncts:
            Sentence.validate(disjunct)
        self.disjuncts = list(disjuncts)

    def __eq__(self, other):
        return isinstance(other, Or) and self.disjuncts == other.disjuncts

    def __hash__(self):
        return hash(
            ("or", tuple(hash(disjunct) for disjunct in self.disjuncts))
        )

    def __repr__(self):
        disjuncts = ", ".join([str(disjunct) for disjunct in self.disjuncts])
        return f"Or({disjuncts})"

    def evaluate(self, model):
        return any(disjunct.evaluate(model) for disjunct in self.disjuncts)

    def formula(self):
        if len(self.disjuncts) == 1:
            return self.disjuncts[0].formula()
        return " ∨ ".join([Sentence.parenthesize(disjunct.formula())
                            for disjunct in self.disjuncts])

    def symbols(self):
        return set.union(*[disjunct.symbols() for disjunct in self.disjuncts])
    
class Xor(Sentence):
    def __init__(self, *operands):
        for operand in operands:
            Sentence.validate(operand)

        self.operands = list(operands)
    
    def __eq__(self, value):
        return isinstance(value, Xor) and self.operands == value.operands
    
    def __hash__(self):
        return hash(
            ("xor", tuple(hash(operand) for operand in self.operands))
        )
    
    def __repr__(self):
        operands = ", ".join([str(operand) for operand in self.operands])
        return f"Xor({operands})"

    def evaluate(self, model):
        return any(operand.evaluate(model) for operand in self.operands) and all(not operand.evaluate(model) for operand in self.operands)

    def formula(self):
        if len(self.operands) == 1:
            return self.operands[0].formula()
        return " ⊕ ".join([Sentence.parenthesize(operand.formula())
                            for operand in self.operands])

    def symbols(self):
        return set.union(*[operand.symbols() for operand in self.operands])


class Implication(Sentence):
    def __init__(self, antecedent, consequent):
        Sentence.validate(antecedent)
        Sentence.validate(consequent)
        self.antecedent = antecedent
        self.consequent = consequent

    def __eq__(self, other):
        return (isinstance(other, Implication)
                and self.antecedent == other.antecedent
                and self.consequent == other.consequent)

    def __hash__(self):
        return hash(("implies", hash(self.antecedent), hash(self.consequent)))

    def __repr__(self):
        return f"Implication({self.antecedent}, {self.consequent})"

    def evaluate(self, model):
        return ((not self.antecedent.evaluate(model))
                or self.consequent.evaluate(model))

    def formula(self):
        antecedent = Sentence.parenthesize(self.antecedent.formula())
        consequent = Sentence.parenthesize(self.consequent.formula())
        return f"{antecedent} => {consequent}"

    def symbols(self):
        return set.union(self.antecedent.symbols(), self.consequent.symbols())


class Biconditional(Sentence):
    def __init__(self, left, right):
        Sentence.validate(left)
        Sentence.validate(right)
        self.left = left
        self.right = right

    def __eq__(self, other):
        return (isinstance(other, Biconditional)
                and self.left == other.left
                and self.right == other.right)

    def __hash__(self):
        return hash(("biconditional", hash(self.left), hash(self.right)))

    def __repr__(self):
        return f"Biconditional({self.left}, {self.right})"

    def evaluate(self, model):
        return ((self.left.evaluate(model)
                 and self.right.evaluate(model))
                or (not self.left.evaluate(model)
                    and not self.right.evaluate(model)))

    def formula(self):
        left = Sentence.parenthesize(str(self.left))
        right = Sentence.parenthesize(str(self.right))
        return f"{left} <=> {right}"

    def symbols(self):
        return set.union(self.left.symbols(), self.right.symbols())


def model_check(knowledge, query):
    """Checks if knowledge base entails query."""

    def check_all(knowledge, query, symbols, model):
        """Checks if knowledge base entails query, given a particular model."""

        # If model has an assignment for each symbol
        if not symbols:

            # If knowledge base is true in model, then query must also be true
            if knowledge.evaluate(model):
                return query.evaluate(model)
            return True
        else:

            # Choose one of the remaining unused symbols
            remaining = symbols.copy()
            p = remaining.pop()

            # Create a model where the symbol is true
            model_true = model.copy()
            model_true[p] = True

            # Create a model where the symbol is false
            model_false = model.copy()
            model_false[p] = False

            # Ensure entailment holds in both models
            return (check_all(knowledge, query, remaining, model_true) and
                    check_all(knowledge, query, remaining, model_false))

    # Get all symbols in both knowledge and query
    symbols = set.union(knowledge.symbols(), query.symbols())

    # Check that knowledge entails query
    return check_all(knowledge, query, symbols, dict())

def move_negation(andor: And|Or) -> Or|And:
    if isinstance(andor, And):
        return Or(Not(s) for s in andor.conjuncts)
    else:
        return And(Not(s) for s in andor.disjuncts)

def distribute_or(disj: Or) -> Or|And:
    def expand_inner_ors(outer_disjuncts):
        disjuncts = []

        for d in outer_disjuncts:
            if isinstance(d, Or):
                disjuncts += expand_inner_ors(d.disjuncts)
            else:
                disjuncts.append(d)
        
        return disjuncts

    def prod(seqs):
        if not seqs:
            return [[]]
        else:
            result = []
            for x in seqs[0]:                       
                for p in prod(seqs[1:]):
                    result.append([x]+p)
            return result

    disjuncts = expand_inner_ors(disj.disjuncts) 

    if any([isinstance(s, And) for s in disjuncts]):
        ls = [[s] if isinstance(s, Symbol) or isinstance(s, Not) else s.conjuncts for s in disjuncts]
        conjuncts = prod(ls)
        conjuncts = [Or(*s) if len(s) > 1 else s for s in conjuncts]
        return And(*conjuncts)
    else:
        return Or(*disjuncts)

    return 
        

def convert_cnf(sentence: Sentence) -> Sentence:
    """
    Converts sentence into conjunctive normal form.
    """
    """
        1. p <=> q = p => q && q => p
        2. p => q = !p || q
        3. !(p &| q) = !p |& !q
        4. p || (q && r) = (p || q) && (p || r)
    """
    if isinstance(sentence, Symbol):
        return sentence
    
    if isinstance(sentence, Biconditional):
        left = convert_cnf(sentence.left)
        right = convert_cnf(sentence.right)

        sentence = And(Implication(left, right), Implication(right, left))        

    if isinstance(sentence, Implication):
        antecedent = convert_cnf(sentence.antecedent)
        consequent = convert_cnf(sentence.consequent)

        sentence = Or(Not(antecedent), consequent)
    
    if isinstance(sentence, Xor):
        operands = [convert_cnf(operand) for operand in sentence.operands]
        sentence = And(Or(*operands), Not(And(*operands)))

    if isinstance(sentence, Not):
        operand = convert_cnf(sentence.operand)

        if isinstance(operand, Symbol):
            return Not(operand)
        
        if isinstance(operand, And):
            print('op', operand.conjuncts)
            sentence = Or(*[Not(s) for s in operand.conjuncts])
        elif isinstance(operand, Or):
            sentence = And(*[Not(s) for s in operand.disjuncts])
        else:
            pass    # unreachable

    if isinstance(sentence, Or):
        sentence = distribute_or(sentence)
    elif isinstance(sentence, And):
        sentence.conjuncts = [convert_cnf(c) for c in sentence.conjuncts]
    
    return sentence

