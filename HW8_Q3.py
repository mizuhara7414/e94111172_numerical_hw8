import numpy as np
from scipy.integrate import quad

print("第三題：離散最小平方三角多項式")
print("="*50)

# 給定函數 f(x) = x² sin(x) 在區間 [0,1]，使用 m = 16
def f(x):
    return x**2 * np.sin(x)

m = 16
print(f"函數：f(x) = x² sin(x)")
print(f"區間：[0, 1]")
print(f"m = {m}")
print()

# 建立離散點
x_discrete = np.linspace(0, 1, m+1)  # m+1 個點，包含端點
y_discrete = f(x_discrete)


# 第一部分：計算離散最小平方三角多項式 S₁(x)
print("3a. 計算離散最小平方三角多項式 S₁(x)")
print("-"*50)

# S₁(x) = a₀ + a₁cos(πx) + b₁sin(πx)
# 基函數：φ₀(x) = 1, φ₁(x) = cos(πx), φ₂(x) = sin(πx)

# 建立設計矩陣 A
A = np.zeros((m+1, 3))
A[:, 0] = 1  # 常數項
A[:, 1] = np.cos(np.pi * x_discrete)  # cos(πx)
A[:, 2] = np.sin(np.pi * x_discrete)  # sin(πx)



coeffs = np.linalg.lstsq(A, y_discrete, rcond=None)[0]
a0, a1, b1 = coeffs

print(f"a₀ = {a0:.8f}")
print(f"a₁ = {a1:.8f}")
print(f"b₁ = {b1:.8f}")
print()
print(f"S₁(x) = {a0:.8f} + {a1:.8f}cos(πx) + {b1:.8f}sin(πx)")

# 定義 S₁(x)
def S1(x):
    return a0 + a1*np.cos(np.pi*x) + b1*np.sin(np.pi*x)

# 計算離散誤差
y_pred = S1(x_discrete)
discrete_error = np.sqrt(np.sum((y_discrete - y_pred)**2))
print(f"離散 L2 誤差：{discrete_error:.8f}")
print()

# 第二部分：計算 ∫₀¹ S₁(x) dx
print("3b. 計算 ∫₀¹ S₁(x) dx")
print("-"*30)

# 解析計算積分
# ∫₀¹ S₁(x) dx = ∫₀¹ [a₀ + a₁cos(πx) + b₁sin(πx)] dx
# = a₀∫₀¹ 1 dx + a₁∫₀¹ cos(πx) dx + b₁∫₀¹ sin(πx) dx
# = a₀·1 + a₁·[sin(πx)/π]₀¹ + b₁·[-cos(πx)/π]₀¹
# = a₀ + a₁·0 + b₁·[-(-1-1)/π]
# = a₀ + b₁·(2/π)



# 數值驗證
integral_S1_numerical, _ = quad(S1, 0, 1)
print(f"∫₀¹ S₁(x) dx = {integral_S1_numerical:.8f}")
print()

# 第三部分：與 ∫₀¹ x² sin(x) dx 比較
print("3c. 與 ∫₀¹ x² sin(x) dx 比較")
print("-"*35)

# 計算原函數的積分
integral_f, _ = quad(f, 0, 1)
print(f"∫₀¹ x² sin(x) dx = {integral_f:.8f}")
print(f"∫₀¹ S₁(x) dx = {integral_S1_numerical:.8f}")
print(f"積分誤差：{abs(integral_f - integral_S1_numerical):.8f}")
print()

# 第四部分：計算連續 L2 誤差 E(S4)
print("3d. 計算連續 L2 誤差 E(S4)")
print("-"*35)

def error_function(x):
    return (f(x) - S1(x))**2

continuous_error_squared, _ = quad(error_function, 0, 1)
continuous_error = np.sqrt(continuous_error_squared)

print(f"E(S₁) = √(∫₀¹ [f(x) - S₁(x)]² dx) = {continuous_error:.8f}")
print()












