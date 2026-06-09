# Micrograd
A manual micro grad framework for forward and backward pass.


Project A: Micro-Grad Framework from Scratch (Foundational)

Instead of using PyTorch or TensorFlow, build a tiny scalar-valued autograd engine from scratch using pure Python.

What you will do: Implement a Value class that stores a number, its gradient, and pointers to the operations that created it. Write the backward pass manually for operations like addition, multiplication, and activation functions ($ReLU$, $Sigmoid$).

Why it helps: You will truly understand how the chain rule propagates gradients backward through a computational graph.

Backend Twist: Expose your tiny library via a FastAPI endpoint where users can send a custom mathematical expression, and your backend returns the forward evaluation and the partial derivatives.
