from sklearn import svm 
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
x , y = samples_generator.make_classification(n_informative=5 , n_redundant=0 , random_state=42)
anova_filter  = SelectKBest(f_regression , k=5)
clf = svm.SVC(kernel ='linear')
anova_svm =Pipeline([('anova' , anova_filter ) , ('svc',clf)])
anova_svm.set_params(anova_k=10 , svc_C=.1).fit(x,y)


prediction = anova_svm.predict(x)
anova_svm.score(x,y)
anova_svm.named_steps['anova'].get_support()
anova_svm.named_steps.anova.get_support()
print(prediction)
