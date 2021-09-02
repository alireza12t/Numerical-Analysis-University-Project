# -*- coding: utf-8 -*-
"""پروژه ۱ مبانی آنالیز عددی.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CV8DFuN2vbC3-RUHpojkavhT9amAx1IT
"""

#run this cell to make text-cells rtl
from IPython.core.display import HTML
HTML("""
<style>
div.text_cell_render{
direction:rtl;
}
</style>
""")

"""**اعضای گروه :**

***امیرحسین داستانی ۹۶۱۳۴۰۸***

***یاسمن امی ۹۶۱۳۰۰۵***

***علیرضا طغیانی ۹۶۱۳۴۱۶***
"""

pip install sympy

"""**اینجا از لایبری سیمپای بهره بردیم که در حل معادلات به ما کمک میکند**"""

from sympy import symbols, Eq, solve, diff, solveset, Interval, sympify

"""تابع زیر با بررسی معادله و دامنه و برد آن شرایط نقطه ثابت رو در معادله بررسی کرده و می‌گوید آیا شرایط نقطه ثابت برقرار بوده یا خیر """

def checkFixedPoint(eq, a, b):
  x = symbols('x')
  f_a = eq.evalf(subs = {x : a})
  f_b = eq.evalf(subs = {x : b})
  dev = diff(eq, x)
  dev_0 = solve(dev, x)
  points = [a, b]
  domain = Interval(a, b)

  for i in dev_0:
    if domain.contains(i):
      points.append(i)
      
  cp = []
  for p in points :
    cp.append(eq.evalf(subs = {x : p}))

  range = Interval(min(cp), max(cp))


  if domain.contains(min(cp)) and domain.contains(max(cp)):
    dev2 = diff(dev, x)
    points = [a, b]
    for p in solve(dev2, x):
      if domain.contains(p):
        points.append(p)
    
    cp = []
    for p in points:
      cp.append(dev.evalf(subs = {x : p}))
    
    min_cp = min(cp)
    max_cp = max(cp)

    if min_cp > -1 and min_cp < 1 and max_cp > -1 and max_cp < 1:
      return True
    else:
      return False 
  else:
    return False

def solver(eqStr, a, b):
  x = symbols('x')
  eq = sympify(eqStr)
  if checkFixedPoint(eq, a, b):
    x0, n = map(float, input("Enter X_0 and number of repetitions:").split(" "))
    points = [x0]
    for i in range(int(n)):
      points.append(eq.evalf(subs = {x : points[-1]}))
    return points
  else:
    print("ERROR => The input function does not have a fixed point theorem condition")

"""برای محاسبه، به تابع ‍`solver` به ترتیب معادله،نقطه‌ی ابتدا و نقطه‌ی انتهای بازه را ورودی دهید"""

solver("1/3*x**2-1/3", -1, 1)