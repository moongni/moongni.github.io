---
layout: single
title: "[Algorithm] 적대적 탐색"
categories: algorithm
tags: [python, algorithm]
classes: wide
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---
# 고전 알고리즘 인 파이썬

<div class = "notice--success">
    <h3> 목차 </h3>
    <a href="https://moongni.github.io/algorithm/Adversarial-Search/#%EC%A0%81%EB%8C%80%EC%A0%81-%ED%83%90%EC%83%89">적대적 탐색</a>
    <ul>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#81-%EB%B3%B4%EB%93%9C-%EA%B5%AC%EC%84%B1">8.1 보드 구성</a>
        </li>
    </ul>
    <a href="https://moongni.github.io/algorithm/Adversarial-Search/#82-%ED%8B%B1%ED%83%9D%ED%86%A0">틱택토</a>
    <ul>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#821-%ED%8B%B1%ED%83%9D%ED%86%A0-%EC%83%81%ED%83%9C-%EA%B4%80%EB%A6%AC">8.2.1 틱택토 상태관리</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#822-%EC%B5%9C%EC%86%8C%EC%B5%9C%EB%8C%80minimax-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98">8.2.1 최소최대 알고리즘</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#823-%ED%8B%B1%ED%83%9D%ED%86%A0-ai">8.2.3 틱택토 AI</a>
        </li>
    </ul>
    <a href="https://moongni.github.io/algorithm/Adversarial-Search/#83-%EC%BB%A4%EB%84%A5%ED%8A%B8%ED%8F%AC">커넥트포</a>
    <ul>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#831-%EC%BB%A4%EB%84%A5%ED%8A%B8%ED%8F%AC-%EA%B5%AC%ED%98%84">8.3.1 커넥트포 구현</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#832-%EC%BB%A4%EB%84%A5%ED%8A%B8%ED%8F%AC-ai">8.3.2 커넥트포 AI</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#833-%EC%95%8C%ED%8C%8C-%EB%B2%A0%ED%83%80-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0%EB%A1%9C-%EC%B5%9C%EC%86%8C%EC%B5%9C%EB%8C%80-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EA%B0%9C%EC%84%A0%ED%95%98%EA%B8%B0">8.3.2 알파-베타 가지치기</a>
        </li>
    </ul>
    <a href="https://moongni.github.io/algorithm/Adversarial-Search/#%EC%97%B0%EC%8A%B5%EB%AC%B8%EC%A0%9C">연습문제</a>
    <ul>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#1-%ED%8B%B1%ED%83%9D%EB%8F%84%EC%97%90-%EB%8B%A8%EC%9C%84-%ED%85%8C%EC%8A%A4%ED%8A%B8%EB%A5%BC-%EC%B6%94%EA%B0%80%ED%95%98%EC%97%AC-legal_moves-is_win-is_draw-%EC%86%8D%EC%84%B1%EC%9D%B4-%EC%9E%98-%EC%9E%91%EB%8F%99%ED%95%98%EB%8A%94%EC%A7%80-%ED%99%95%EC%9D%B8%ED%95%98%EB%9D%BC">1번</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#2-%EC%BB%A4%EB%84%A5%ED%8A%B8%ED%8F%AC%EC%97%90-%EB%8C%80%ED%95%9C-%EC%B5%9C%EC%86%8C%EC%B5%9C%EB%8C%80-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EC%9D%98-%EB%8B%A8%EC%9C%84-%ED%85%8C%EC%8A%A4%ED%8A%B8%EB%A5%BC-%EC%9E%91%EC%84%B1%ED%95%98%EB%9D%BC">2번</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#3-tictactoe_aipy%EC%99%80-connectfour_aipy%EC%9D%98-%EC%BD%94%EB%93%9C%EB%A5%BC-%EB%91%90-%EA%B2%8C%EC%9E%84-%EB%AA%A8%EB%91%90%EC%97%90%EC%84%9C-%EC%82%AC%EC%9A%A9%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8F%84%EB%A1%9D-%EB%91%90-%EB%A9%94%EC%84%9C%EB%93%9C%EB%A5%BC-%EC%9E%91%EC%84%B1%ED%95%98%EC%97%AC-%EB%A6%AC%ED%8C%A9%ED%86%A0%EB%A7%81%ED%95%98%EB%9D%BC">3번</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Adversarial-Search/#4-%EC%BB%B4%ED%93%A8%ED%84%B0-%ED%94%8C%EB%A0%88%EC%9D%B4%EC%96%B4%EA%B0%80-%EC%9E%90%EC%8B%A0%EA%B3%BC-%EA%B2%8C%EC%9E%84%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8F%84%EB%A1%9D-connectfour_aipy-%EC%BD%94%EB%93%9C%EB%A5%BC-%EB%B3%80%EA%B2%BD%ED%95%B4%EB%B3%B4%EC%9E%90-%EC%B2%AB-%EB%B2%88%EC%A7%B8%EC%99%80-%EB%91%90-%EB%B2%88%EC%A7%B8-%ED%94%8C%EB%A0%88%EC%9D%B4%EC%96%B4%EC%9D%98-%EC%8A%B9%EB%A5%A0%EC%9D%80">4번</a>
        </li>
    </ul>
</div>

## 적대적 탐색
2인용 게임, 제로섬, 완전 정보 게임인 틱택토, 커넥트포, 체커스, 체스 등과 같은 게임의 인공적인 상대 플레이어를 만든느 법을 배운다.  
  
### 8.1 보드 구성
탐색 알고리즘을 게임별로 지정하지 않고 기본 클래스를 정의하여 구현한다. 이후 구현하고자 하는 게임에서 기본 클래스를 서브클래스로 상속받고, 탐색 알고리즘을 사용한다.  

**Board** 추상 클래스에서 게임의 상태를 관리한다.  
탐색 알고리즘이 계산할 게임에 대해 알아야될 속성  
- 누구 turn인가?
- 말은 현재 위치에서 어디로 움직일 수 있는가?
- 이겼는가?
- 무승부인가?
무승부는 두 번째과 세 번째 질문의 조합이다.

나머지 두 조치
- 현재 위치에서 새 위치로 이동한다.
- 플레이어의 말 위치를 평가하여 어느 쪽이 유리한지 평가한다.


```python
# board.py
from __future__ import annotations
from typing import NewType, List
from abc import ABC, abstractmethod

Move = NewType('Move', int) # 말이 놓일 보드 List의 인덱스(정수) 타입

class Piece:
    @property
    def opposite(self) -> Piece:
        raise NotImplementedError("서브클래스로 구현해야 합니다.")
        
class Board(ABC):
    @property
    @abstractmethod
    def turn(self) -> Piece:
        ...
        
    @property
    @abstractmethod
    def legal_moves(self) -> List[Move]:
        ...
    
    @property
    @abstractmethod
    def is_win(self) -> bool:
        ...
    
    @property
    def is_draw(self) -> bool:
        return (not self.is_win) and (len(self.legal_moves) == 0)

    @abstractmethod
    def move(self, location: Move) -> Board:
        ...
   
    @abstractmethod
    def evaluate(self, player: Piece) -> float:
        ...
        
```

## 8.2 틱택토
### 8.2.1 틱택토 상태 관리
틱텍토 보드의 각 위치는 1차원 리스트로 아래와 같은 형식이다.
```
0|1|2
-----
3|4|5
-----
6|7|8
```
**TTTBoard** 클래스는 게임 상대를 저장하며, 서로 다른 두말의 상태를 추적하며 틱택토의 말은 **X** , **O** , 빈공간(E)으로 표시한다.

**NOTE**: 틱택토의 게임 규칙 [https://ko.wikipedia.org/wiki/틱택토](https://ko.wikipedia.org/wiki/%ED%8B%B1%ED%83%9D%ED%86%A0)
{: .notice--info}


```python
#tictactoe.py
from __future__ import annotations
from typing import List
from enum import Enum
from board import Piece, Board, Move


class TTTPiece(Piece, Enum):
    X = "X"
    O = "O"
    E = " " # 보드가 초기화 되는 빈 공간
    
    # 틱택토 게임에서 한 플레이어가 말을 이동한 후 다른 플레이어 턴으로 넘겨준다
    @property
    def opposite(self) -> TTTPiece:
        if self == TTTPiece.X:
            return TTTPiece.O
        elif self == TTTPiece.O:
            return TTTPiece.X
        else:
            return TTTPiece.E
    
    def __str__(self) -> str:
        return self.value
    
class TTTBoard(Board):
    def __init__(self, position: List[TTTPiece] = [TTTPiece.E] * 9, turn: TTTPiece = TTTPiece.X) -> None:
        self.position: List[TTTPiece] = position
        self._turn: TTTPiece = turn
    
    # 말을 놓을 수 있는 차례
    @property
    def turn(self) -> Piece:
        return self._turn
    
    # 말을 이동한다.
    def move(self, location: Move) -> Board:
        temp_position: List[TTTPiece] = self.position.copy()
        temp_position[location] = self._turn
        return TTTBoard(temp_position, self._turn.opposite)
    
    @property
    def legal_moves(self) -> List[Move]:
        return [Move(l) for l in range(len(self.position)) if self.position[l] == TTTPiece.E]
    
    @property
    def is_win(self) -> bool:
        # 3행, 3열, 2개의 대각선을 확인한다.
        row_1: bool = self.position[0] != TTTPiece.E and self.position[0] == self.position[1] \
            and self.position[0] == self.position[2] # 1행
        row_2: bool = self.position[3] != TTTPiece.E and self.position[3] == self.position[4] \
            and self.position[3] == self.position[5] # 2행
        row_3: bool = self.position[6] != TTTPiece.E and self.position[6] == self.position[7] \
            and self.position[6] == self.position[8] # 3행
        col_1: bool = self.position[0] != TTTPiece.E and self.position[0] == self.position[3] \
            and self.position[0] == self.position[6] # 1열
        col_2: bool = self.position[1] != TTTPiece.E and self.position[1] == self.position[4] \
            and self.position[1] == self.position[7] # 2열
        col_3: bool = self.position[2] != TTTPiece.E and self.position[2] == self.position[5] \
            and self.position[2] == self.position[8] # 3열
        dia_1: bool = self.position[0] != TTTPiece.E and self.position[0] == self.position[4] \
            and self.position[0] == self.position[8] # \ 대각선
        dia_2: bool = self.position[2] != TTTPiece.E and self.position[2] == self.position[4] \
            and self.position[2] == self.position[6] # / 대각선
            
        return row_1 or row_2 or row_3 or col_1 or col_2 or col_3 or dia_1 or dia_2
    
    def evaluate(self, player: Piece) -> float:
        if self.is_win and self.turn == player:
            return -1 # 지는 경우
        elif self.is_win and self.turn != player:
            return 1  # 이기는 경우
        else:
            return 0  # 비기는 경우
        
    def __repr__(self) -> str:
        return f"""{self.position[0]}|{self.position[1]}|{self.position[2]}
-----
{self.position[3]}|{self.position[4]}|{self.position[5]}
-----
{self.position[6]}|{self.position[7]}|{self.position[8]}"""
```

<hr>

### 8.2.2 최소최대(minimax) 알고리즘
최소최대는 완벽한 정보를 가진 2인용 제로섬 게임에서 최적의 이동을 찾는 고전 알고리즘이다. 최소최대 알고리즘은 각 플레이어가 최대화 플레이어 또는 최소화 플레이어로 지정된 재귀함수를 사용하여 구현한다.  
   
- **최대화 플레이어** : 최대이익을 얻을 수 있는 이동을 목표로 한다.  
  
최대화 플레이어의 이득을 최대화하려는 시도 후 재귀적으로 호출되어 상대방 최대화 플레이어의 이득을 최소화하는 이동을 찾는다.  
이를 재귀 함수의 기저 조건에 도달할 때까지 반복한다.  
  
- **기저 조건** : 게임 종료(승리 또는 무승부) 또는 최대 깊이 탐색

**TTTBoard** 클래스의 `evaluate()`메서드에서 프레이어가 최대화 플레이어에게 이기면 1점, 지면 -1점, 비기면 0점을 얻는다.  
  
이 점수는 기저 조건에 도달하면 반환되는데, 이 기저 조건에 연결된 모든 재귀 호출을 통해 점수가 버블링된다. 최대화 작업에서 각 재귀 호출은 최고 평가를 위해 한 단계 더 버블링한다. 반대로 최소화 작업에서 각 재귀호출은 최저 평가를 위해 한 단계 더 버블링한다. 이런 방식으로 **의사 결정 트리**가 작성된다.  
  
탐색 공간이 너무 커서 게임 종료 위치에 도달할 수 없는 게임의 경우(체커, 체스와 같은 경우) 최소최대 알고리즘은 특정 깊이(탐색을 위한 말의 이동 깊이 수, 플라이 라고도 함)까지 탐색 후 중단된다. 그런 다음 휴리스틱을 사용하여 게임 상태를 평가한다.  


```python
# minimax.py
from __future__ import annotations
from board import Piece, Board, Move


# 게임 플레이어의 가능한 최선의 움직임을 찾는다.
def minimax(board: Board, maximizing: bool, original_player: Piece, max_depth: int = 8) -> float:
    # 기저 조건 - 게임 종료 위치 또는 최대 깊이에 도달한다.
    if board.is_win or board.is_draw or max_depth == 0:
        return board.evaluate(original_player)

    # 재귀 조건 - 이익을 극대화하거나 상대방의 이익을 최소화한다.
    if maximizing:
        best_eval: float = float("-inf") # 낮은 시작 점수
        for move in board.legal_moves:
            result: float = minimax(board.move(move), False, original_player, max_depth - 1)
            best_eval = max(result, best_eval) # 가장 높은 평가를 받은 위치로 움직인다.
        return best_eval
    else: # 최소화
        worst_eval: float = float("inf") # 높은 시작 점수
        for move in board.legal_moves:
            result = minimax(board.move(move), True, original_player, max_depth - 1)
            worst_eval = min(result, worst_eval) # 가장 낮은 평가를 받은 위치로 움직인다.
        return worst_eval

# 헬퍼 함수를 생성하여 각 유효한 이동에 대해
# minimax() 함수 호출을 반복하여 가장 높은 값으로 평가되는 이동을 찾는다.
# 최대 깊이(max_depth) 전까지 최선의 움직임을 찾는다.
def find_best_move(board: Board, max_depth: int = 8) -> Move:
    best_eval: float = float("-inf")
    best_move: Move = Move(-1)
    for move in board.legal_moves:
        result: float = minimax(board.move(move), False, board.turn, max_depth)
        if result > best_eval:
            best_eval = result
            best_move = move
    return best_move
```

<hr>

### 8.2.3 틱택토 AI
다음 코드 조각에서 AI 는 먼저 말을 두는 인간을 상대로 플레이 한다.


```python
#tictactoe_ai.py
from minimax import find_best_move
from tictactoe import TTTBoard
from board import Move, Board

board: Board = TTTBoard()


def get_player_move() -> Move:
    player_move: Move = Move(-1)
    while player_move not in board.legal_moves:
        play: int = int(input("이동할 위치를 입력하세요 (0-8):"))
        player_move = Move(play)
    return player_move


if __name__ == "__main__":
    # main game loop
    while True:
        human_move: Move = get_player_move()
        board = board.move(human_move)
        if board.is_win:
            print("당신이 이겼습니다!")
            break
        elif board.is_draw:
            print("비겼습니다!")
            break
        computer_move: Move = find_best_move(board)
        print(f"컴퓨터가 {computer_move}(으)로 이동했습니다.")
        board = board.move(computer_move)
        print(board)
        if board.is_win:
            print("컴퓨터가 이겼습니다!")
            break
        elif board.is_draw:
            print("비겼습니다!")
            break
```

    이동할 위치를 입력하세요 (0-8):4
    컴퓨터가 0(으)로 이동했습니다.
    O| | 
    -----
     |X| 
    -----
     | | 

    ...

    이동할 위치를 입력하세요 (0-8):1
    컴퓨터가 7(으)로 이동했습니다.
    O|X|X
    -----
    X|X|O
    -----
    O|O| 
    이동할 위치를 입력하세요 (0-8):8
    비겼습니다!
    

위에서 최소최대 알고리즘의 `max_depth`의 값이 8이므로 컴퓨터플레이어는 항상 게임의 마지막 수까지 생각한다.  
따라서 매 게임마다 완벽하게 알고리즘이 동작하고 컴퓨터플레이어를 이길 수 없다.  


## 8.3 커넥트포
커넥트포는 세워져 있는 7 * 6 격자판에 두 명의 플레이어가 교대로 말을 두어 가로, 세로 또는 대각선으로 연속적인 4개를 만들면 이기는 게임이다. 보드가 완전히 채워지면 무승부가 된다.  
  
<hr>

### 8.3.1 커넥트포 구현

```python
# connectfour.py
from __future__ import annotations
from typing import List, Optional, Tuple
from enum import Enum
from board import Piece, Board, Move

class C4Piece(Piece, Enum):
    B = "B"
    R = "R"
    E = " " # 빈 공간
    
    @property
    def opposite(self) -> C4Piece:
        if self == C4Piece.B:
            return C4Piece.R
        elif self == C4Piece.R:
            return C4Piece.B
        else:
            return C4Piece.E
        
    def __str__(self) -> str:
        return self.value
```


```python
# connectfour.py
...

def generate_segments(num_columns: int, num_rows: int, segment_length: int) -> List[List[Tuple[int, int]]]:
    segments: List[List[Tuple[int, int]]] = []
    # 수직 세그먼트 생성
    for c in range(num_columns):
        for r in range(num_rows - segment_length + 1):
            segment: List[Tuple[int, int]] = []
            for t in range(segment_length):
                segment.append((c, r + t))
            segments.append(segment)
    
    # 수평 세그먼트 생성
    for r in range(num_rows):
        for c in range(num_columns - segment_length + 1):
            segment: List[Tuple[int, int]] = []
            for t in range(segment_length):
                segment.append((c + t, r))
            segments.append(segment)
    
    # 왼쪽 아래에서 오른쪽 위 대각선의 세그먼트
    for c in range(num_columns - segment_length + 1):
        for r in range(num_rows - segment_length + 1):
            segment = []
            for t in range(segment_length):
                segment.append((c + t, r + t))
            segments.append(segment)
            
    # 왼쪽 위에서 오른쪽 아래 대각선의 세그먼트
    for c in range(num_columns - segment_length + 1):
        for r in range(segment_length - 1, num_rows):
            segment = []
            for t in range(segment_length):
                segment.append((c + t, r - t))
            segments.append(segment)
    return segments
```

- **세그먼트** : 4개의 격자위치 리스트  
보드의 세그먼트가 모두 같은 색의 말이면 해당 색의 플레이어가 게임에서 이긴 것이다.


```python
# connectfour.py
...

class C4Board(Board):
    NUM_COLUMNS: int = 7
    NUM_ROWS: int = 6
    SEGMENT_LENGHT: int = 4
    SEGMENTS: List[List[Tuple[int, int]]] = generate_segments(NUM_COLUMNS, NUM_ROWS, SEGMENT_LENGHT)
    
    # 내부 클래스
    class Column:
        def __init__(self) -> None:
            self._container: List[C4Piece] = []

        @property
        def full(self) -> bool:
            return len(self._container) == C4Board.NUM_ROWS

        def push(self, item: C4Piece) -> None:
            if self.full:
                raise OverflowError("격자 열 범위에 벗어날 수 없습니다.")
            self._container.append(item)

        def __getitem__(self, index: int) -> C4Piece:
            if index > len(self._container) - 1:
                return C4Piece.E
            return self._container[index]

        def __repr__(self) -> str:
            return repr(self._container)

        def copy(self) -> C4Board.Column:
            temp: C4Board.Column = C4Board.Column()
            temp._container = self._container.copy()
            return temp
        
    def __init__(self, position: Optional[List[C4Board.Column]] = None,
                turn: C4Piece = C4Piece.B) -> None:
        if position is None:
            self.position: List[C4Board.Column] = [C4Board.Column() for _ in range(C4Board.NUM_COLUMNS)]
        else:
            self.position = position
        self._turn: C4Piece = turn
        
    @property
    def turn(self) -> Piece:
        return self._turn
    
    def move(self, location: Move) -> Board:
        temp_position: List[C4Board.Column] = self.position.copy()
        for c in range(C4Board.NUM_COLUMNS):
            temp_position[c] = self.position[c].copy()
        temp_position[location].push(self._turn)
        return C4Board(temp_position, self._turn.opposite)
    
    @property
    def legal_moves(self) -> List[Move]:
        return [Move(c) for c in range(C4Board.NUM_COLUMNS) if not self.position[c].full]
    
    # 세그먼트의 검은 말과 빨간 말의 수를 반환한다.
    # 같은 색 말이 4개가 있는 세그먼트가 있는지 확인하여 승리를 결정한다.
    def _count_segment(self, segment: List[Tuple[int, int]]) -> Tuple[int, int]:
        black_count: int = 0
        red_count: int = 0
        for column, row in segment:
            if self.position[column][row] == C4Piece.B:
                black_count += 1
            elif self.position[column][row] == C4Piece.R:
                red_count += 1
        return black_count, red_count
    
    @property
    def is_win(self) -> bool:
        for segment in C4Board.SEGMENTS:
            black_count, red_count = self._count_segment(segment)
            if black_count == 4 or red_count == 4:
                return True
        return False
    
    def _evaluate_segment(self, segment: List[Tuple[int, int]], player: Piece) -> float:
        black_count, red_count = self._count_segment(segment)
        if red_count > 0 and black_count > 0:
            return 0 # 말이 혼합된 세그먼트 0 점
        count: int = max(red_count, black_count)
        score: float = 0
        if count == 2:
            score = 1
        elif count == 3:
            score = 100
        elif count == 4:
            score = 1000000
        color: C4Piece = C4Piece.B
        if red_count > black_count:
            color = C4Piece.R
        if color != player:
            return -score
        return score
    
    def evaluate(self, player: Piece) -> float:
        total: float = 0
        for segment in C4Board.SEGMENTS:
            total += self._evaluate_segment(segment, player)
        return total
    
    def __repr__(self) -> str:
        display: str = ""
        for r in reversed(range(C4Board.NUM_ROWS)):
            display += "|"
            for c in range(C4Board.NUM_COLUMNS):
                display += f"{self.position[c][r]}" + "|"
            display += "\n"
        return display
```

- **Column** 클래스는 내부 클래스로 커넥트포 보드를 7개의 열 그룹으로 생각하여 개념적으로 쉽게 이해하기 위해 사용한다.  
Stack 클래스와 비슷한 형태를 가지나 `pop`되지 않는 스택이다.  
  
- **`__getitem__()` 특수 메서드를 사용해 Column 인스턴스를 인덱싱할 수 있어서 열 리스트를 2차원 리스트처럼 취급할 수 있다.**
  
**CAUTION: 위에 틱택토에서 처럼 1차원 또는 2차원 리스트로 격자를 표현하는 것이 성능적으로 더 좋을 수 있다.**
{: .notice--danger}

- **C4Board** 클래스 또한 수정없이 **Board** 추상 클래스의 `is_draw` 속성을 사용할 수 있다.  
  
- `_evaluate_segment()` 는 단일 세그먼트를 평가하는 헬퍼 메서드이다.
    - 말이 섞여 있는 경우 : 0점
    - 같은 색 말 2개, 빈 공간 2개 : 1점
    - 같은 색 말 3개, 빈 공간 1개 : 100점
    - 같은 색 말 4개 : 1000000점 (승리)
  
`evaluate()` 메서드는 `_evaluate_segment()` 메서드를 사용하여 모든 세그먼트의 총 점수를 반환한다.

<hr>

### 8.3.2 커넥트포 AI
위에 틱택토에서 사용한 `minimax()` 와 `find_best_move()` 함수를 같이 적용할 수 있다.  
`max_depth` 변수를 3으로 설정하여 컴퓨터 말의 동작당 사고 시간을 합리적으로 만들어 준다.  


```python
from minimax import find_best_move
from connectfour import C4Board
from board import Move, Board

board: Board = C4Board()


def get_player_move() -> Move:
    player_move: Move = Move(-1)
    while player_move not in board.legal_moves:
        play: int = int(input("이동할 열 위치를 입력하세요 (0-6):"))
        player_move = Move(play)
    return player_move


if __name__ == "__main__":
    # 메인 게임 루프
    while True:
        human_move: Move = get_player_move()
        board = board.move(human_move)
        if board.is_win:
            print("당신이 이겼습니다!")
            break
        elif board.is_draw:
            print("비겼습니다!")
            break
        computer_move: Move = find_best_move(board, 3)
        print(f"컴퓨터가 {computer_move} 열을 선택했습니다.")
        board = board.move(computer_move)
        print(board)
        if board.is_win:
            print("컴퓨터가 이겼습니다!")
            break
        elif board.is_draw:
            print("비겼습니다!")
            break
```

    이동할 열 위치를 입력하세요 (0-6):1
    컴퓨터가 3 열을 선택했습니다.
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | |B| |R| | | |

    ...
    
    이동할 열 위치를 입력하세요 (0-6):5
    컴퓨터가 3 열을 선택했습니다.
    | | | | | | | |
    | |R| |R| | | |
    | |B| |R|R|B| |
    | |R| |R|B|R| |
    | |R| |B|B|B| |
    | |B| |R|B|B| |
    
    이동할 열 위치를 입력하세요 (0-6):2
    당신이 이겼습니다!
    

<hr>

### 8.3.3 알파-베타 가지치기로 최소최대 알고리즘 개선하기
최소최대 알고리즘은 잘 작동하지만 매우 깊은 탐색은 할 수 없다. **알파-베타 가지치기**는 이미 탐색한 위치보다 점수가 낮은 탐색 위치를 제외시켜 알고리즘의 탐색 깊이를 향상시킬 수 있다.  
  
**알파**는 탐색 트리에서 현재까지 발견된 최고의 최대화 움직임 평가를 나타낸다.  
**베타**는 상대방에 대해 현재까지 발견된 최고의 최소화 움직임 평가를 나타낸다.  

> 베타 <= 알파: 해당 위치 분기에서 발견될 위치보다 더 좋거나 같은 위치가 이미 발견되었기 때문에 넘어간다.  
휴리스틱을 통해 탐색공간을 줄인다.


```python

def alphabeta(board: Board, maximizing: bool, original_player: Piece, max_depth: int = 8, alpha: float = float("-inf"), beta: float = float("inf")) -> float:
    # 기저 조건 - 게임 종료 위치 또는 최대 깊이에 도달한다.
    if board.is_win or board.is_draw or max_depth == 0:
        return board.evaluate(original_player)

    # 재귀 조건 - 이익을 극대화하거나 상대방의 이익을 최소화한다.
    if maximizing:
        for move in board.legal_moves:
            result: float = alphabeta(board.move(move), False, original_player, max_depth - 1, alpha, beta)
            alpha = max(result, alpha)
            if beta <= alpha:
                break
        return alpha
    else:  # 최소화
        for move in board.legal_moves:
            result = alphabeta(board.move(move), True, original_player, max_depth - 1, alpha, beta)
            beta = min(result, beta)
            if beta <= alpha:
                break
        return beta

```

## 연습문제

### 1. 틱택도에 단위 테스트를 추가하여 `legal_moves`, `is_win`, `is_draw` 속성이 잘 작동하는지 확인하라


```python
import unittest
from typing import List
from minimax import find_best_move
from tictactoe import TTTPiece, TTTBoard
from board import Move

class TTTMethodTest(unittest.TestCase):
    def test_legal_moves(self):
        move_positions: List[TTTPiece] = [TTTPiece.X, TTTPiece.O, TTTPiece.X,
                                         TTTPiece.X, TTTPiece.X, TTTPiece.O,
                                         TTTPiece.O, TTTPiece.E, TTTPiece.O]
        test_board1: TTTBoard = TTTBoard(move_positions, TTTPiece.X)
        answer1: List[Move] = test_board1.legal_moves
        self.assertEqual(answer1, [7])
    
    def test_is_win(self):
        win_positions: List[TTTPiece] = [TTTPiece.X, TTTPiece.O, TTTPiece.X,
                                         TTTPiece.X, TTTPiece.O, TTTPiece.O,
                                         TTTPiece.X, TTTPiece.E, TTTPiece.O]
        test_board2: TTTBoard = TTTBoard(win_positions, TTTPiece.X)
        answer2: bool = test_board2.is_win
        self.assertTrue(answer2)
    
    def test_is_draw(self):
        draw_positions: List[TTTPiece] = [TTTPiece.X, TTTPiece.O, TTTPiece.X,
                                         TTTPiece.X, TTTPiece.O, TTTPiece.O,
                                         TTTPiece.O, TTTPiece.X, TTTPiece.O]
        test_board3: TTTBoard = TTTBoard(draw_positions, TTTPiece.X)
        answer3: bool = test_board3.is_draw
        self.assertTrue(answer3)
        
if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```

    ...
    ----------------------------------------------------------------------
    Ran 3 tests in 0.007s
    
    OK
    
<hr>

### 2. 커넥트포에 대한 최소최대 알고리즘의 단위 테스트를 작성하라

```python
import unittest
from typing import List, Tuple
from minimax import find_best_move
from connectfour import C4Piece, C4Board
from board import Move

class ConnectfourMinimaxTest(unittest.TestCase):
    def test_easy_position(self):
        # 1번 움직여서 이기는 자리 찾기
        column1: C4Board.Column = C4Board.Column()
        
        column2: C4Board.Column = C4Board.Column()
        column2.push(C4Piece.R)
        column2.push(C4Piece.B)
        
        column3: C4Board.Column = C4Board.Column()
        column3.push(C4Piece.B)
        column3.push(C4Piece.R)
        column3.push(C4Piece.B)
        column3.push(C4Piece.R)
        column3.push(C4Piece.B)

        column4: C4Board.Column = C4Board.Column()
        column4.push(C4Piece.B)
        column4.push(C4Piece.B)
        column4.push(C4Piece.R)
        column4.push(C4Piece.R)
        column4.push(C4Piece.B)
        
        column5: C4Board.Column = C4Board.Column()
        column5.push(C4Piece.R)
        column5.push(C4Piece.R)
        column5.push(C4Piece.B)
        
        column6: C4Board.Column = C4Board.Column()

        column7: C4Board.Column = C4Board.Column()
        
        to_win_easy_position: List[List[C4Piece]] = []
        to_win_easy_position.append(column1)
        to_win_easy_position.append(column2)
        to_win_easy_position.append(column3)
        to_win_easy_position.append(column4)
        to_win_easy_position.append(column5)
        to_win_easy_position.append(column6)
        to_win_easy_position.append(column7)
        
        test_board1: C4Board = C4Board(to_win_easy_position, C4Piece.R)
        answer1: Move = find_best_move(test_board1, 3)
        self.assertEqual(answer1, 4)
    
    def test_block_position(self):
        # B가 우승하는 것을 막기
        column1: C4Board.Column = C4Board.Column()
        column1.push(C4Piece.R)
        
        column2: C4Board.Column = C4Board.Column()
        column2.push(C4Piece.B)
        column2.push(C4Piece.R)
        
        column3: C4Board.Column = C4Board.Column()
        column3.push(C4Piece.B)
        column3.push(C4Piece.B)

        column4: C4Board.Column = C4Board.Column()
        column4.push(C4Piece.R)
        column4.push(C4Piece.B)
        column4.push(C4Piece.R)
        column4.push(C4Piece.R)

        column5: C4Board.Column = C4Board.Column()
        column5.push(C4Piece.B)
        column5.push(C4Piece.R)
        column5.push(C4Piece.B)
     
        column6: C4Board.Column = C4Board.Column()
        column6.push(C4Piece.R)
        column6.push(C4Piece.B)
        column6.push(C4Piece.B)
        
        column7: C4Board.Column = C4Board.Column()
        
        to_block_position: List[List[C4Piece]] = []
        to_block_position.append(column1)
        to_block_position.append(column2)
        to_block_position.append(column3)
        to_block_position.append(column4)
        to_block_position.append(column5)
        to_block_position.append(column6)
        to_block_position.append(column7)
        
        test_board2: C4Board = C4Board(to_block_position, C4Piece.R)
        answer2: Move = find_best_move(test_board2, 3)
        self.assertEqual(answer2, 5)
    
    def test_hard_position(self):
        # 2번을 고려해야 하는 어려운 경우
        column1: C4Board.Column = C4Board.Column()
        column1.push(C4Piece.B)
       
        column2: C4Board.Column = C4Board.Column()
        column2.push(C4Piece.B)
        column2.push(C4Piece.R)

        column3: C4Board.Column = C4Board.Column()
        column3.push(C4Piece.B)
        column3.push(C4Piece.B)
        column3.push(C4Piece.R)
        column3.push(C4Piece.B)
        
        column4: C4Board.Column = C4Board.Column()
        column4.push(C4Piece.R)
        column4.push(C4Piece.B)
        column4.push(C4Piece.R)
        column4.push(C4Piece.R)
        
        column5: C4Board.Column = C4Board.Column()
        column5.push(C4Piece.B)
        column5.push(C4Piece.R)
        column5.push(C4Piece.B)
        
        column6: C4Board.Column = C4Board.Column()
        column6.push(C4Piece.R)
        
        column7: C4Board.Column = C4Board.Column()
        
        to_win_hard_position: List[List[C4Piece]] =[]
        to_win_hard_position.append(column1)
        to_win_hard_position.append(column2)
        to_win_hard_position.append(column3)
        to_win_hard_position.append(column4)
        to_win_hard_position.append(column5)
        to_win_hard_position.append(column6)
        to_win_hard_position.append(column7)
        
        test_board3: C4Board = C4Board(to_win_hard_position, C4Piece.R)
        answer3: Move = find_best_move(test_board3, 3)
        self.assertEqual(answer3, 3)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)    
```

    ...
    ----------------------------------------------------------------------
    Ran 3 tests in 3.678s
    
    OK

<hr>

### 3. tictactoe_ai.py와 connectfour_ai.py의 코드를 두 게임 모두에서 사용할 수 있도록 두 메서드를 작성하여 리팩토링하라


```python
from __future__ import annotations
from typing import NewType, List
from abc import ABC, abstractmethod

Move = NewType('Move', int)


class Piece:
    @property
    def opposite(self) -> Piece:
        raise NotImplementedError("Should be implemented by subclasses.")


class Board(ABC):
    @property
    @abstractmethod
    def turn(self) -> Piece:
        ...

    @abstractmethod
    def move(self, location: Move) -> Board:
        ...

    @property
    @abstractmethod
    def legal_moves(self) -> List[Move]:
        ...

    @property
    @abstractmethod
    def is_win(self) -> bool:
        ...

    @property
    def is_draw(self) -> bool:
        return (not self.is_win) and (len(self.legal_moves) == 0)

    @abstractmethod
    def evaluate(self, player: Piece) -> float:
        ...
```


```python
from __future__ import annotations
from typing import List, Optional, Tuple
from enum import Enum
from board import Piece, Board, Move

def generate_segments(num_columns: int, num_rows: int, segment_length: int) -> List[List[int]]:
    segments: List[int] = []
    # 열 세그먼트
    for c in range(num_columns):
        for r in range(num_rows - segment_length + 1):
            segment: List[int] = []
            for t in range(segment_length):
                segment.append(c + (r + t) * num_columns)
            segments.append(segment)
    # 행 세그먼트
    for r in range(num_rows):
        for c in range(num_columns - segment_length + 1):
            segment: List[int] = []
            for t in range(segment_length):
                segment.append(c + t + r * num_columns)
            segments.append(segment)
    # / 대각선 세그먼트
    for c in range(num_columns - segment_length + 1):
        for r in range(num_rows - segment_length + 1):
            segment = []
            for t in range(segment_length):
                segment.append(c + r * num_columns + t * (num_columns + 1))
            segments.append(segment)
    # \ 대각선 세그먼트
    for c in range(num_columns - segment_length + 1):
        for r in range(segment_length - 1, num_rows):
            segment = []
            for t in range(segment_length):
                segment.append(c + r * num_columns - t * (num_columns - 1))
            segments.append(segment)
            
    return segments

class GamePiece(Piece, Enum):
    F = "F" # first player
    S = "S" # second player
    E = " " # 빈 공간
    
    @property
    def opposite(self) -> GamePiece:
        if self == GamePiece.F:
            return GamePiece.S
        elif self == GamePiece.S:
            return GamePiece.F
        else:
            return GamePiece.E
    
    def __str__(self) -> str:
        return self.value
    
class GameBoard(Board):
    GameType = Enum("GameType", "tictactoe connectfour")
    
    def __init__(self, position: List[GamePiece] = None, game_type: GameType = GameType.tictactoe, turn: GamePiece = GamePiece.F) -> None:
        self.game_type = game_type
        if position == None:
            if self.game_type == GameBoard.GameType.tictactoe:
                self.position = [GamePiece.E] * 9
                self.num_columns = 3
                self.num_rows = 3
                self.segment_length = 3

            elif self.game_type == GameBoard.GameType.connectfour:
                self.position = [GamePiece.E] * 42
                self.num_columns = 7
                self.num_rows = 6
                self.segment_length = 4
        else:
            if self.game_type == GameBoard.GameType.tictactoe:
                self.num_columns = 3
                self.num_rows = 3
                self.segment_length = 3

            elif self.game_type == GameBoard.GameType.connectfour:
                self.num_columns = 7
                self.num_rows = 6
                self.segment_length = 4

            self.position = position
            
        self.segments = generate_segments(self.num_columns, self.num_rows, self.segment_length)
        self._turn: GamePiece = turn
    
    @property
    def turn(self) -> Piece:
        return self._turn

    def move(self, location: Move) -> Board:
        temp_position: List[GamePiece] = self.position.copy()
        if self.game_type == GameBoard.GameType.tictactoe:
            temp_position[location] = self._turn
        else:
            temp_location = location % 7
            while self.position[location] != GamePiece.E:
                location += 7
            temp_position[location] = self._turn
        return GameBoard(temp_position, self.game_type, self._turn.opposite)
    
    @property
    def legal_moves(self) -> List[Move]:
        if self.game_type == GameBoard.GameType.tictactoe:
            return [Move(t) for t in range(len(self.position)) if self.position[t] == GamePiece.E]
        else:
            return list(set(Move(c % 7) for c in range(len(self.position)) if self.position[c] == GamePiece.E))
    
    def _count_segment(self, segment: List[int]) -> Tuple[int, int]:
        first_count: int = 0
        second_count: int = 0
        for i in segment:
            if self.position[i] == GamePiece.F:
                first_count += 1
            elif self.position[i] == GamePiece.S:
                second_count += 1
        return first_count, second_count
    
    @property
    def is_win(self) -> bool:
        for segment in self.segments:
            first_count, second_count = self._count_segment(segment)
            if first_count == self.segment_length or second_count == self.segment_length:
                return True
        return False
    
    def _evaluate_segment(self, segment: List[int], player: Piece) -> float:
        first_count, second_count = self._count_segment(segment)
        if first_count > 0 and second_count > 0:
            return 0 # 완성 불가
        count: int = max(first_count, second_count)
        score: float = 0
        if count == 2:
            score = 1
        if count == 3:
            score = 1000
        if count == 4:
            score = 1000000
        color: GamePiece = GamePiece.F
        if first_count < second_count:
            color = GamePiece.S
        if color != player:
            return -score
        return score
    
    def evaluate(self, player: Piece) -> float:
        total: float = 0
        for segment in self.segments:
            total += self._evaluate_segment(segment, player)
        return total
    
    def __repr__(self) -> str:
        display: str = ""
        for r in reversed(range(self.num_rows)):
            display += "|"
            for c in range(self.num_columns):
                display += f"{self.position[c + r * self.num_columns]}" + "|"
            display += "\n"
        return display
```

커넥트포 게임의 경우 위에서 **Column** 내부클래스를 사용한 것과 달리 리스트를 통해 게임을 진행할 수 있도록 메서드를 바꿨다. 커넥트포 보드의 형태는 아래와 같다.
```
|35|36|37|38|39|40|41|
|28|29|30|31|32|33|34|
|21|22|23|24|25|26|27|
|14|15|16|17|18|19|20|
|7 |8 |9 |10|11|12|13|
|0 |1 |2 |3 |4 |5 |6 |
```

<hr>

### 4. 컴퓨터 플레이어가 자신과 게임할 수 있도록 connectfour_ai.py 코드를 변경해보자. 첫 번째와 두 번째 플레이어의 승률은?


```python
from minimax import find_best_move
from connectfour import C4Board
from board import Move, Board
from random import randrange

if __name__ == "__main__":
    # main game loop

    for move in range(7):
        first_win = 0
        second_win = 0

        for _ in range(5):
            board: Board = C4Board()
            board = board.move(move)

            while True:
                second_move: Move = find_best_move(board, 3)
                board = board.move(second_move)
                if board.is_win:
                    print("second wins!")
                    second_win += 1
                    break
                elif board.is_draw:
                    print("Draw!")
                    break
                first_move: Move = find_best_move(board, 3)
                board = board.move(first_move)
                if board.is_win:
                    print("first wins!")
                    first_win += 1
                    break
                elif board.is_draw:
                    print("Draw!")
                    break
        print(f"{move}이동을 시작으로 5번을 실행한 결과 first : {first_win} second : {second_win}")
```

    ...

    0이동을 시작으로 5번을 실행한 결과 first : 5 second : 0

    ...

    1이동을 시작으로 5번을 실행한 결과 first : 0 second : 5

    ...

    2이동을 시작으로 5번을 실행한 결과 first : 0 second : 5

    ...

    3이동을 시작으로 5번을 실행한 결과 first : 5 second : 0

    ...

    4이동을 시작으로 5번을 실행한 결과 first : 0 second : 5

    ...

    5이동을 시작으로 5번을 실행한 결과 first : 5 second : 0

    ...

    6이동을 시작으로 5번을 실행한 결과 first : 5 second : 0
    

게임을 바로 시작하면 first 컴퓨터가 `find_best_move()` 메서드로 인해 Move(2)를 선택하는 경우가 나와 second 컴퓨터가 이기는 경우만 나오게 된다.  
first의 첫 move를 변화시키며 5번씩 경기를 진행했을 때, first : second 가 4 : 3 의 비율로 승리하는 것을 볼 수 있다.
