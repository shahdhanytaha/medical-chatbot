from rag import retriever

query = "What are the symptoms of stroke?"

docs = retriever.invoke(query)

print("\nQUERY:")
print(query)

for i, doc in enumerate(docs, 1):

    print("\n" + "=" * 60)
    print("DOCUMENT:", i)
    print("SOURCE:", doc.metadata.get("source"))
    print("CONTENT:")
    print(doc.page_content)