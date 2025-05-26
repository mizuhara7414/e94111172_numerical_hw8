import numpy as np
import matplotlib.pyplot as plt

# 給定數據
x_data = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y_data = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])



print(f"x: {x_data}")
print(f"y: {y_data}")
print()

# 1a. 二次多項式
print("1a. 二次多項式 (y = ax² + bx + c)")
print("-"*50)


n = len(x_data)
A_quad = np.vstack([x_data**2, x_data, np.ones(n)]).T
coeffs_quad = np.linalg.lstsq(A_quad, y_data, rcond=None)[0]

a, b, c = coeffs_quad
print(f"係數：a = {a:.6f}, b = {b:.6f}, c = {c:.6f}")
print(f"二次逼近：y = {a:.6f}x² + {b:.6f}x + {c:.6f}")


y_pred_quad = a * x_data**2 + b * x_data + c
error_quad = np.sqrt(np.sum((y_data - y_pred_quad)**2))
print(f"誤差：{error_quad:.6f}")
print()

# 1b. 指數形式
print("1b. 指數形式 (y = be^(ax))")
print("-"*50)


ln_y = np.log(y_data)
A_exp = np.vstack([x_data, np.ones(n)]).T
coeffs_exp = np.linalg.lstsq(A_exp, ln_y, rcond=None)[0]

a_exp, ln_b = coeffs_exp
b_exp = np.exp(ln_b)
print(f"係數：a = {a_exp:.6f}, b = {b_exp:.6f}")
print(f"指數逼近：y = {b_exp:.6f} * e^({a_exp:.6f}x)")

# 計算誤差
y_pred_exp = b_exp * np.exp(a_exp * x_data)
error_exp = np.sqrt(np.sum((y_data - y_pred_exp)**2))
print(f"誤差：{error_exp:.6f}")
print()

# 1c. 冪函數)
print("1c. 冪函數形式最小平方逼近 (y = bx^a)")
print("-"*50)


ln_x = np.log(x_data)
ln_y = np.log(y_data)
A_power = np.vstack([ln_x, np.ones(n)]).T
coeffs_power = np.linalg.lstsq(A_power, ln_y, rcond=None)[0]

a_power, ln_b_power = coeffs_power
b_power = np.exp(ln_b_power)
print(f"係數：a = {a_power:.6f}, b = {b_power:.6f}")
print(f"冪函數逼近：y = {b_power:.6f} * x^{a_power:.6f}")

# 計算誤差
y_pred_power = b_power * (x_data ** a_power)
error_power = np.sqrt(np.sum((y_data - y_pred_power)**2))
print(f"誤差：{error_power:.6f}")
print()




