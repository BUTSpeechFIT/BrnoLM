import io

def getStream(words):
    data_source = io.StringIO()
    data_source.write(" ".join(words))
    data_source.seek(0)

    return data_source
