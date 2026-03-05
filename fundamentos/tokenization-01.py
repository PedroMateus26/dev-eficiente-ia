import nltk

nltk.download("punkt_tab")

text = "Machine learning (aprendizado de máquina) é um subconjunto da inteligência artificial (IA) que capacita sistemas computacionais a aprender e melhorar autonomamente por meio de dados. Sem programação explícita. Algoritmos analisam grandes volumes de dados para identificar padrões, tomar decisões e aprimorar seu desempenho com o tempo à medida que são treinados."

word_tokens = nltk.word_tokenize(text)

print(word_tokens)

sentence_tokens = nltk.sent_tokenize(text)

print(sentence_tokens)
