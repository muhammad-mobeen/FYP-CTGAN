from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from starlette.responses import FileResponse, HTMLResponse
from io import BytesIO
import pandas as pd
from mangum import Mangum
from ctgan import CTGAN


app = FastAPI()
handler = Mangum(app)

def run_ctgan(data, smpl):
    categorical_features = ['age','sex','children','smoker','region']
    ctgan = CTGAN(verbose=True)
    ctgan.fit(data, categorical_features, epochs = 200)
    return ctgan.sample(smpl)

# HTML form to collect user input
html_form = """
<!doctype html>
<html>
  <head>
    <title>CSV Processing</title>
  </head>
  <body>
    <h2>CSV Processing Form</h2>
    <form action="/process_csv" method="post" enctype="multipart/form-data">
      <label for="sample_amount">Enter Sample Amount:</label>
      <input type="number" name="sample_amount" required>
      <br>
      <label for="csv_file">Upload CSV File:</label>
      <input type="file" name="csv_file" accept=".csv" required>
      <br>
      <button type="submit">Process CSV</button>
    </form>
  </body>
</html>
"""

@app.get("/")
async def serve_form():
    return HTMLResponse(content=html_form, status_code=200)

@app.post("/process_csv")
async def process_csv(
    sample_amount: int = Form(...),
    csv_file: UploadFile = File(...)
):
    # Read the uploaded CSV file
    content = await csv_file.read()
    df = pd.read_csv(BytesIO(content))

    sample_df = run_ctgan(df, sample_amount)

    # Perform some operations on the CSV data
    # For example, here we just multiply a column by the sample_amount
    # df['Value'] = df['Value'] * sample_amount

    # Generate a new CSV file as BytesIO
    # output_csv = BytesIO()
    # df.to_csv(output_csv, index=False)

    # Prepare the new CSV file for download
     # Save the updated DataFrame to a new CSV file
    output_csv = "output.csv"
    sample_df.to_csv(output_csv, index=False)
    
    # Return the new CSV file to the user for download
    return FileResponse(output_csv, media_type='application/octet-stream', headers={'Content-Disposition': f'attachment; filename="{output_csv}"'})
                                            