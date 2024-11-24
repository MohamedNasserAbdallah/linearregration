import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# تحميل البيانات
dataset = pd.read_csv("data/Salary_Data.csv")

# تقسيم البيانات
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# تقسيم البيانات إلى تدريب واختبار
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

# إنشاء نموذج الانحدار الخطي
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# التنبؤ بالقيم
y_pred = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)

# طباعة القيم الفعلية والمتنبأ بها
print(pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
}))

# رسم البيانات
plt.scatter(x_train, y_train, color="red", label="Actual (Training Data)")
plt.plot(x_train, y_train, color="green", label="Actual Line (Training Data)")  # رسم الخط الفعلي
plt.plot(x_train, y_pred_train, color="blue", label="Predicted Line (Training Data)")  # رسم خط التنبؤ للبيانات التدريبية
plt.plot(x_test, y_pred, color="pink", label="Predicted Line (Test Data)")  # رسم خط التنبؤ للبيانات الاختبارية

# إضافة العناوين والتسميات
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

# إضافة الأسطورة
plt.legend()

# عرض الرسم البياني
plt.show()
