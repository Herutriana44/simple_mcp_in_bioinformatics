import asyncio
from datetime import datetime
from google import genai
from google.genai import types
from admet_ai import ADMETModel  # pip install admet-ai
from Bio.Seq import Seq  # pip install biopython
from Bio import Blast  # pip install biopython>=1.85
import os

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

client = genai.Client(api_key=GEMINI_API_KEY)

# Tidak ada MCP server eksternal khusus bioinformatika, jadi None
server_params = None

# PROMPT = f"Lakukan BLAST search untuk dna ATGCCGATG menggunakan blastn dan database nr dengan email herutriana44@gmail.com"
PROMPT = f"Lakukan analisis ADMET dengan senyawa SMILES O(c1ccc(cc1)CCOC)CC(O)CNC(C)C"

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

async def run():
    if server_params is not None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = [
                    types.Tool(
                        function_declarations=[tool]
                    ) for tool in bioinfo_tools
                ]
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=PROMPT,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        tools=tools,
                    ),
                )
                if response.candidates[0].content.parts[0].function_call:
                    function_call = response.candidates[0].content.parts[0].function_call
                    print(f"Function call: {function_call}")
                    if function_call.name == "run_blast":
                        result = run_blast(**function_call.args)
                        print(f"Tool Result: {result}")
                    elif function_call.name == "predict_admet":
                        result = predict_admet(**function_call.args)
                        print(f"Result of predict_admet:    ")
                        for key, value in result.items():
                            print(f"{key}: {value}")
                    elif function_call.name == "dna_to_protein":
                        result = dna_to_protein(**function_call.args)
                        print(f"Tool Result: {result}")
                    else:
                        result = await session.call_tool(function_call.name, arguments=function_call.args)
                        print(f"Tool Result: {result}")
                else:
                    print("No function call found in the response.")
                    print(response.text)
    else:
        # Jalankan tanpa MCP server eksternal, hanya tools lokal
        tools = [
            types.Tool(
                function_declarations=[tool]
            ) for tool in bioinfo_tools
        ]
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=PROMPT,
            config=types.GenerateContentConfig(
                temperature=0,
                tools=tools,
            ),
        )
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            print(f"Function call: {function_call}")
            if function_call.name == "run_blast":
                result = run_blast(**function_call.args)
                print(f"Tool Result: {result}")
            elif function_call.name == "predict_admet":
                result = predict_admet(**function_call.args)
                print(f"Result of predict_admet:")
                result_string = ""
                for key, value in result.items():
                    result_string += f"{key}: {value}\n"
                print(result_string)
                # prompt to gemini
                prompt_admet_result = f"Lakukan analisis hasil ADMET berikut: \n {result_string}"
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt_admet_result,
                )
                print(response.text)
            elif function_call.name == "dna_to_protein":
                result = dna_to_protein(**function_call.args)
                print(f"Tool Result: {result}")
            else:
                result = f"Tool {function_call.name} not available."
                print(f"Tool Result: {result}")
        else:
            print("No function call found in the response.")
            print(response.text)

asyncio.run(run()) 
