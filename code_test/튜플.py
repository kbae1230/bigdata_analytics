a = "{{2},{2,1},{2,1,3},{2,1,3,4}}"

def solution(s):
    result = []
    srcs = sorted(s[2:-2].split('},{'), key = lambda x : len(x))
    for src in srcs:
        result += list(set(map(int, src.split(','))) - set(result))
    
    return result

print(solution(a))