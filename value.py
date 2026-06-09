import math

class Value:
    def __init__(self, data, children = () , op = '') -> None:
        self.data = data
        self.children = children
        self.op = op
        self.label = ""
        self.grad = 0.0
        self._backward = lambda : None
        
    def add_label(self, label):
        self.label = label
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data , (self,other), '+' )
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
        
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        out = Value(self.data * other.data , (self,other), '*' )
        
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward
        return out
        

    def __rmul__(self, other):
        return self * other
    
    def exp(self):
        o = math.exp(2*self.data) - 1 / math.exp(2*self.data) + 1
        out = Value(o, children=(self,), op="tanh")
        def _backward():
            self.grad += (1-(o**2)) * out.grad
            
        out._backward = _backward
        return out
    
    def backward(self):
        vis = set()
        topo = []
        
        def build_topo(node):
            if node not in vis:
                vis.add(node)
            
            for child in node.children:
                build_topo(child)
            
            topo.append(node)
        build_topo(self)
        
        self.grad = 1.0
        
        for ele in reversed(topo):
            ele._backward()
        