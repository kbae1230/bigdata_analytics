# 클래스 요소 : 멤버변수 ( 필드 , 속성 ), 멤버메소드(기능), 생성자(필수)
# __init__ 기본 생성자
# 생성자도 매개변수가 있는 생성자가 가능

class Car:
    color = ''
    spped = 0

    def __init__(self, color, speed) -> None:
        self.color = color
        self.speed = speed

myCar1 = Car(color = '빨간', speed = 30)
