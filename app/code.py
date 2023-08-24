import numpy as np
carBrand ={
    0:"Adui",
    1:'Hyundai Creta',
    2:'Mahindra Scorpio',
    3:'Rolls Royce',
    4:'Swift',
    5:'Tata Safari',
    6:'Toyota Innova'
}

def predict_Car(model,hog):
    brand = model.predict(np.array(hog).reshape(1,-1))
    return carBrand[brand[0]]
    