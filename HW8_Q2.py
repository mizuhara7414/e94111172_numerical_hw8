import numpy as np
from scipy.integrate import quad



# 定義函數 f(x) = (1/2)cos(x) + (1/4)sin(2x) 在區間 [-1,1]
def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2*x)

print("f(x) = (1/2)cos(x) + (1/4)sin(2x)")

print("find : P₂(x) = a₀ + a₁x + a₂x²")
print()


G = np.zeros((3, 3))

# G[0,0] = ∫₋₁¹ 1·1 dx = 2
G[0,0] = 2.0

# G[0,1] = G[1,0] = ∫₋₁¹ 1·x dx = 0 (奇函數在對稱區間上的積分)
G[0,1] = G[1,0] = 0.0

# G[0,2] = G[2,0] = ∫₋₁¹ 1·x² dx = 2/3
G[0,2] = G[2,0] = 2.0/3.0

# G[1,1] = ∫₋₁¹ x·x dx = 2/3
G[1,1] = 2.0/3.0

# G[1,2] = G[2,1] = ∫₋₁¹ x·x² dx = 0 (奇函數)
G[1,2] = G[2,1] = 0.0

# G[2,2] = ∫₋₁¹ x²·x² dx = 2/5
G[2,2] = 2.0/5.0


b = np.zeros(3)

# b[0] = ∫₋₁¹ f(x)·1 dx
b[0], _ = quad(lambda x: f(x), -1, 1)

# b[1] = ∫₋₁¹ f(x)·x dx
b[1], _ = quad(lambda x: f(x) * x, -1, 1)

# b[2] = ∫₋₁¹ f(x)·x² dx
b[2], _ = quad(lambda x: f(x) * x**2, -1, 1)


# 解正規方程組 Ga = b
coeffs = np.linalg.solve(G, b)
a0, a1, a2 = coeffs

print("最小平方二次多項式逼近：")
print(f"P₂(x) = {a0:.6f} + {a1:.6f}x + {a2:.6f}x²")
print()

# 定義逼近多項式
def P2(x):
    return a0 + a1*x + a2*x**2

# 計算逼近誤差 (L2 範數)
def error_function(x):
    return (f(x) - P2(x))**2

error_squared, _ = quad(error_function, -1, 1)
error = np.sqrt(error_squared)

print(f"L2 誤差 = √(∫₋₁¹ [f(x) - P₂(x)]² dx) = {error:.6f}")
print()
