from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


#LLM
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)






#Tesla text to chunk
tesla_text = """Malaria is a life-threatening disease spread to humans by some types of mosquitoes. It is
mostly found in tropical countries. It is preventable and curable.
The infection is caused by a parasite and does not spread from person to person.
Symptoms can be mild or life-threatening. Mild symptoms are fever, chills and headache.
Severe symptoms include fatigue, confusion, seizures, and difficulty breathing.
Infants, children under 5 years, pregnant women and girls, travellers and people with HIV
or AIDS are at higher risk of severe infection.
Malaria can be prevented by avoiding mosquito bites and with medicines. Treatments can
stop mild cases from getting worse.
Malaria mostly spreads to people through the bites of some infected
female Anopheles mosquitoes. Blood transfusion and contaminated needles may also
transmit malaria. The first symptoms may be mild, similar to many febrile illnesses, and
difficulty to recognize as malaria. Left untreated, P. falciparum malaria can progress to
severe illness and death within 24 hours.
There are 5 Plasmodium parasite species that cause malaria in humans and 2 of these
speciesP. falciparum and P. vivax  pose the greatest threat. P. falciparum is the deadliest
malaria parasite and the most prevalent on the African continent. P. vivax is the dominant
malaria parasite in most countries outside of sub-Saharan Africa. The other malaria species
which can infect humans are P. malariae, P. ovale and P. knowlesi.
."""







#prompt
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

print("AI Chunking...")
res = llm.invoke(prompt)
marked_text = res.content

chunks = marked_text.split("<<<SPLIT>>>")


# Clean up the chunks
clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        clean_chunks.append(cleaned)

# Show results
print("\n🎯 AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()