from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = FastAPI()

app = application

# Configure the templates directory
templates = Jinja2Templates(directory="templates")


# Route for the home page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# Route to display the prediction form
@app.get("/predictdata", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={}
    )


# Route to process the prediction form
@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint(
    request: Request,
    gender: str = Form(...),
    ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):
    # Collect form data
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    # Convert input into a DataFrame
    pred_df = data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    # Load the trained model and predict
    predict_pipeline = PredictPipeline()
    print("Mid Prediction")

    results = predict_pipeline.predict(pred_df)
    print("After Prediction")

    # Display the result in the HTML page
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"results": results[0]}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

