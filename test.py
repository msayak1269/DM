def getResult():
    # input = request.form.to_dict()
    attributes = [[0, 135,  68, 42, 250,  42.3, 0.365,  24]] 
    col = ['Pregnancies', 'Glucose',  'BloodPressure',  'SkinThickness',  'Insulin',  'BMI',  'DiabetesPedigreeFunction', 'Age']
    df1 = pd.DataFrame(attributes, columns = col)
    loaded_model = pickle.load(open("RF_model.pkl", "rb"))
    result = loaded_model.predict(df1)
    print(result[0])

