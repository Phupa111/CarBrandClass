from fastapi import FastAPI,Request
from code import predict_Car
import pickle
import requests

app = FastAPI()
url = "http://172.17.0.2:8080/api/gethog"
m = pickle.load(open(r'model\carBandModel.pk','rb'))

@app.get("/")
def root():
    return {"message": "this is car api"}

@app.get("/api/predictCar")
async def read_img64(request:Request):
    item = await request.json()
    item_img = item["img"] 
    item_img = str.split(str(item_img),",")[1]
    response = requests.get(url,json={"img":item_img})
    hog = response.json()
    brand = predict_Car(m,hog['Hog'])
    return {"Brand":brand}