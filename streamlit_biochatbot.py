import streamlit as st
from google import genai
from google.genai import types
import os
import asyncio
from datetime import datetime
from admet_ai import ADMETModel  # pip install admet-ai
from Bio.Seq import Seq  # pip install biopython
from Bio import Blast  # pip install biopython>=1.85


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC3h0u2Vh9BDAqvodxB7NPwRVROXr4YYNM")
client = genai.Client(api_key=GEMINI_API_KEY)

# --- Bioinformatics Tools ---
def run_blast(program, database, sequence, email=None):
    # program: 'blastn', 'blastp', dst
    # database: 'nt', 'nr', dst
    # sequence: string DNA/protein (bisa FASTA)
    # email: opsional, untuk NCBI
    if email:
        Blast.email = email
    result_stream = Blast.qblast(program, database, sequence)
    # result_stream: bytes, default XML
    # Parsing ringkas: ambil summary alignment dan e-value
    from Bio import SeqIO
    from io import BytesIO
    # Simpan hasil ke file sementara
    with open("blast_result.xml", "wb") as f:
        f.write(result_stream.read())
    # Parsing hasil
    blast_records = Blast.parse("blast_result.xml")
    summary = []
    for record in blast_records:
        for hit in record:
            for hsp in hit:
                summary.append({
                    "target_id": hit.target.id,
                    "target_desc": hit.target.description,
                    "score": hsp.score,
                    "evalue": hsp.annotations.get("evalue"),
                    "identity": hsp.annotations.get("identity"),
                })
    return summary

def predict_admet(smiles):
    predictor = ADMETModel()
    result = predictor.predict(smiles=smiles)
    return result

def dna_to_protein(dna_sequence):
    seq = Seq(dna_sequence)
    return str(seq.translate())

# --- MCP Tool Declarations ---
bioinfo_tools = [
    {
        "name": "run_blast",
        "description": "Run BLAST (online NCBI) for a given sequence string.",
        "parameters": {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "BLAST program (blastn, blastp, etc)"},
                "database": {"type": "string", "description": "NCBI database name (nt, nr, etc)"},
                "sequence": {"type": "string", "description": "DNA/protein sequence (string or FASTA)"},
                "email": {"type": "string", "description": "(Optional) Email for NCBI usage", "default": ""}
            },
            "required": ["program", "database", "sequence"]
        }
    },
    {
        "name": "predict_admet",
        "description": "Predict ADMET properties from SMILES using admet-ai.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string"}
            },
            "required": ["smiles"]
        }
    },
    {
        "name": "dna_to_protein",
        "description": "Convert DNA sequence to protein (amino acid sequence).",
        "parameters": {
            "type": "object",
            "properties": {
                "dna_sequence": {"type": "string", "description": "DNA sequence"}
            },
            "required": ["dna_sequence"]
        }
    }
]

def ask_gemini(user_prompt):
    tools = [
        types.Tool(
            function_declarations=[tool]
        ) for tool in bioinfo_tools
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            tools=tools,
        ),
    )
    # Cek apakah Gemini ingin memanggil tool
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call.name == "run_blast":
            result = run_blast(**function_call.args)
            return f"[BLAST Result]\n{result}"
        elif function_call.name == "predict_admet":
            result = predict_admet(**function_call.args)
            result_string = ""
            for key, value in result.items():
                result_string += f"{key}: {value}\n"
            prompt_admet_result = f"Lakukan analisis hasil ADMET berikut: \n {result_string}"
            response_admet = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_admet_result,
            )
            return f"[ADMET Result]\n{result_string}\n{response_admet.text}"
            
        elif function_call.name == "dna_to_protein":
            result = dna_to_protein(**function_call.args)
            return f"[Protein Result]\n{result}"
        else:
            return f"[Tool {function_call.name} not available]"
    else:
        return response.text

# Streamlit UI
st.set_page_config(page_title="Bioinformatics MCP Chatbot", page_icon="🧬")
st.title("🧬 Bioinformatics MCP Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

user_input = st.chat_input("Ketik pertanyaan bioinformatika...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.spinner("Memproses..."):
        response = ask_gemini(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.experimental_rerun()
