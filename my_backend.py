from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from machine  import translate_large_text

#uncomment thos tow comment if you want to use the model from the training notebook
#Comment 1
"""
import torch
from models.architecture import Encoder, Decoder, Seq2Seqmodel
from models import seq2seq_french_model
from transformers import AutoTokenizer


#tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder()
decoder = Decoder()

my_model = Seq2Seqmodel(encoder, decoder, device).to(device)
#now we load the dic state

# !!!! CRUCIAL: As explicitly demonstrated in the notebook, the custom model 
# yields poor results due to the limited training data available. 
# Make sure to replace "seq2seq_french_model.pt" with your own trained weights.
dic_state = torch.load("models/seq2seq_french_model.pt", map_location=device)
my_model.load_state_dict(dic_state)
my_model.eval()

def costum_translation( input_text, model= my_model, tokenizer= tokenizer):
    tokens_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=64)["input_ids"].to(device)

    model.eval()
    with torch.no_grad():
        output_ids = model(tokens_ids, targ=None, teacher_forcing_ratio=0.0)

    predicted_ids = output_ids.argmax(-1).squeeze(0)

    french_translation = tokenizer.decode(predicted_ids, skip_special_tokens=True)

    return french_translation

"""





#let's create an app istance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
Template = Jinja2Templates(directory="templates" )

#the varible we will follow 
class Translationvarible(BaseModel):
    text: str
    languge_target : str
    model: str

#the home page the user will see
@app.get("/")  
def home_page(my_request: Request):

    
    return Template.TemplateResponse(
        request=my_request, 
        name="index_2.html", 
        context={"request": my_request}
    )

@app.post("/translation")
async def translat_function(data: Translationvarible):
    # Default fallback text if nothing matches
    translated_text = ""

    if data.model =="MarianMT":
        if data.languge_target == "Arabic":
            translated_text = translate_large_text(text= data.text, model_name= "Helsinki-NLP/opus-mt-en-ar" )
        elif data.languge_target == "French": 
            translated_text = translate_large_text(text= data.text, model_name= "Helsinki-NLP/opus-mt-en-fr")   

    else:

        #Comment 2
        """
        translated_text = costum_translation(input_text= data.text)
        """
        translated_text="the Seq2Seq is comming sonne"
    return {"translated_text": translated_text}         


