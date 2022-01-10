def find(word):
    result = False

    while True:
        line = (yield result)
        result = word in line

f = find('python')
next(f)