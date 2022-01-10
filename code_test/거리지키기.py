from collections import deque

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

def dfs(i, j, place):
    visit = [[0]*5 for _ in range(5)]
    
    q = deque()
    q.append(i, j, 0)
    visit[i][j] = 0
    while q:
        x, y, dst = q.popleft()
        if 0 < dst < 3 and place == 'P':
            return False
        if dst >= 3:
            break

        for id in range(4):
            nx, ny, nd = x + dx[id], y + dy[id], dst + 1
            if 0 <= nx < 5 and 0 <= ny < 5:
                if place[nx] != 'X' and not visit[nx][ny]:
                    q.append(nx, ny, nd)
                    visit[nx][ny] = 1

    return True
                 




def solution(places):
    answer = []
    for place in places:
        check = 0
        for i in range(len(place)):
            for j in range(len(place[i])):
                if not dfs(i, j, place):
                    check = 1
                    answer.append(0)
                    break
            if check:
                break
            else:
                answer.append(1)

    return answer
places = [["POOOP", "OXXOX", "OPXPX", "OOXOX", "POXXP"], ["POOPX", "OXPXP", "PXXXO", "OXXXO", "OOOPP"], ["PXOPX", "OXOXP", "OXPOX", "OXXOP", "PXPOX"], ["OOOXX", "XOOOX", "OOOXX", "OXOOX", "OOOOO"], ["PXPXP", "XPXPX", "PXPXP", "XPXPX", "PXPXP"]]
print(solution(places))
