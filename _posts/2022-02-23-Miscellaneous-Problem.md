---
layout: single
title: "[Algorithm] 기타 문제"
categories: algorithm
tags: [python, algorithm]
classes: wide
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---
# 고전 알고리즘 인 파이썬
## 배낭 문제

**배낭 문제**는 한정된 배낭에 주어진 물건을 넣어서 배낭에 담을 수 있는 물건의 최대 이익을 찾는 **조합 최적화** 문제다.  
  
배낭에 무게 제한이 없으면 물건의 가치를 무게로 나눠 가장 가치있는 물건을 가져갈 수 있다. 그러나 현실적으로 무게의 제한이 있으므로 한 물건을 취하지 않거나(0), 취하는 것(1)으로 가정한다.(0/1 배낭문제)  
  
**배낭 75파운드**

물건|무게|가치
---|---|---
TV|50 파운드|500 달러
촛대|2 파운드|300 달러
오디오|35 파운드|400 달러
노트북|3 파운드|1000 달러
식량|15 파운드|50 달러
옷|20 파운드|800 달러
보석|1 파운드|4000 달러
책|100 파운드|300 달러
프린터|18 파운드|30 달러
냉장고|200 파운드|700 달러
그림|10 파운드|1000 달러


```python
# knapsack.py
from typing import NamedTuple, List

class Item(NamedTuple):
    name: str
    weight: int
    value: float
```

배낭 문제와 같은 최적화문제에서 사용할 수 있는 다양한 방식 중 **브루트 포스**는 가방에 담을 수 있는 모든 경우의 조합을 확인하는 것으로 수학적으로 **멱집합**이라고 한다.  
  
$N$ 개의 물건이 있으면 (챙기는 경우 / 안 챙기는 경우) 모든 경우의 수는 $2^N$ 개로 시간복잡도는 $O(2^N)$ 이다.
  
> **만약 물건의 수가 많아진다면 기하급수적으로 조합이 늘어나기 때문에 브루트 포스 방식으로는 해결이 어렵다.**

<hr>

### 동적 계획법
동적 계획법은 더 큰 문제를 구성하는 하위 문제를 해결하고, 저장된 결과를 활용하여 더 큰 문제를 해결한다. 배낭 용량이 별도의 단계로 고려될 때 동적 계획법으로 문제를 해결할 수 있다.  


```python
# knapsack.py
...

def knapsack(items: List[Item], max_capacity: int) -> List[Item]:
    # 동적 계획법 표를 작성
    table: List[List[float]] = [[0.0 for _ in range(max_capacity + 1)] for _ in range(len(items) + 1)]
        
    for i , item in enumerate(items):
        for capacity in range(1, max_capacity + 1):
            previous_items_value: float = table[i][capacity]
            if capacity >= item.weight: # 물건이 배낭 용량보다 작은 경우
                value_freeing_weight_for_item: float = table[i][capacity - item.weight]
                # 이전 물건보다 더 가치가 있는 경우에만 물건을 넣는다.
                table[i + 1][capacity] = max(value_freeing_weight_for_item + item.value, previous_items_value)
            else: # 물건이 배낭 용량보다 큼
                table[i + 1][capacity] = previous_items_value

    # 작성된 표에서 최상의 결과를 구한다.
    solution: List[Item] = []
    capacity = max_capacity
    for i in range(len(items), 0, -1): # 거꾸로 진행
        # 배낭에 물건이 있는가
        if table[i - 1][capacity] != table[i][capacity]:
            solution.append(items[i - 1])
            # 용량에서 물건 무게를 뺀다.
            capacity -= items[i - 1].weight
    return solution

# ===== 실행 =====
if __name__ == "__main__":
    items: List[Item] = [
                        Item("보석", 1, 4000),
                        Item("촛대", 2, 300),
                        Item("노트북", 3, 1000),
                        Item("그림", 10, 1000),
                        Item("식량", 15, 50),
                        Item("프린터", 18, 30),
                        Item("옷", 20, 800),
                        Item("오디오", 35, 400),
                        Item("TV", 50, 500),
                        Item("책", 100, 300),
                        Item("냉장고", 200, 700)]
    bags = knapsack(items, 75)
    totalWeight = 0
    totalValue = 0
    for item in bags:
        totalWeight += item.weight
        totalValue += item.value
        print(f"item: {item.name} \t weight: {item.weight} \t \
        value: {item.value} \t totalW: {totalWeight} \t totalV: {totalValue}")
```

    item: 오디오 	 weight: 35 	         value: 400 	 totalW: 35 	 totalV: 400
    item: 옷 	 weight: 20 	         value: 800 	 totalW: 55 	 totalV: 1200
    item: 그림 	 weight: 10 	         value: 1000 	 totalW: 65 	 totalV: 2200
    item: 노트북 	 weight: 3 	         value: 1000 	 totalW: 68 	 totalV: 3200
    item: 촛대 	 weight: 2 	         value: 300 	 totalW: 70 	 totalV: 3500
    item: 보석 	 weight: 1 	         value: 4000 	 totalW: 71 	 totalV: 7500
    

- `knapsack` 함수를 살펴보면 첫 번째 반복문은 $N * C$ 번 수행된다. $N$은 물건의 수, $C$는 배낭의 최대용량이다. 따라서 시간복잡도는 $O(N * C)$ 이다.  
- 가능한 물건의 수에 대해 배낭의 최대 용량까지 반복한다. `i = 2` 인 경우 처음 물건과 두 번째 물건의 조합으로 가져갈 수 있는 최대 가치를 탐색한다.  
- `previous_items_value` 변수는 현재까지 탐색한 물건의 무게의 합이며 새로운 물건을 추가 할 수 있는지 확인한다.(if)  
- `value_freeing_weight_for_item` (새 물건을 넣은 값)이 이전 탐색한 무게보다 더 큰 경우 `table`에 할당하고 더 적은 경우 이전 탐색한 무게를 할당한다.  

`table`은 아래와 같은 형식으로 작성되게 된다.  

|table | 0 파운드 | 1 파운드 | 2 파운드 | 3 파운드 | 4파운드 |  $\cdots$ | 75 파운드|
|---|---|---|---|---|---|---|---|---|
|보석 | 0 | 4000 | 4000 | 4000 | 4000 | $\cdots$ | 4000|
|촛대 | 0 | 4000 | 4000 | 4300 | 4300 | $\cdots$ | 4300|
|노트북 | 0 | 4000 | 4000 | 4300 | 5000 | $\cdots$ | 5300|
|$\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\ddots$ | $\vdots$ |
|냉장고 | 0 | 4000 | 4000 | 4300 | 5000 | $\cdots$ | 7500 |

- `solution` 은 table의 오른쪽 아래에서부터 삽입된 값에 변화가 있었는지 확인하여 변화가 있다면 특정 조합에서 새 물건이 추가 됬다는 것을 의미한다. 배낭에서 새 물건의 무게를 감소하고 반복한다.  

> **브루트 포스 $2^{11}(=2048)$ 회 실행에 비해 동적 계획법 $11*75(=825)$ 회 실행으로 효율적이다.**
  

## 외판원 문제
외판원 문제는 시작 도시에서 끝 도시까지 여러 도시를 한 번에 방문해야 한다. 모든 도시는 서로 직접 연결되어 있으며, 외판원은 도시를 순차적으로 방문한다. 외판원이 모든 도시를 방문하는 최단 경로를 찾는 것이다.  
  
이전 포스팅의 [최소신장트리](https://moongni.github.io/algorithm/Graph-Problems/#442-%EC%B5%9C%EC%86%8C-%EC%8B%A0%EC%9E%A5-%ED%8A%B8%EB%A6%AC-%EC%B0%BE%EA%B8%B0)는 모든 도시를 연결하는 가장 짧은 경로지만, 모든 도시를 한번만 방문하는 가장 짧은 경로는 제공하지 않는다.  
  
외판원 문제는 **NP-난해**다. 알고리즘의 실행 시간은 입력 크기의 다항식 함수이다. NP-난해는 다항 시간에 풀 수 있는 알고리즘이 존재하지 않는다. 외판원이 방문해야 하는 도시 수가 증가할 수록 문제를 풀기 어려워진다. 현재 수백만 개의 도시에서 외판원 문제를 완벽하게 합리적인 시간 내에 해결하는 것은 불가능하다.  
  
<hr>
예시로 외판원이 버몬트 주의 5개 도시를 방문한다. 출발 도시는 정해지지 않았으며, 아래의 표는 5개의 도시와 도시 간의 거리를 보여준다.  

|-|러틀랜드|벌링턴|화이트 리버 정션|베닝턴|브래틀보로|
|---|:---:|:---:|:---:|:---:|:---:|
|러틀랜드|0|67|46|55|75|
|벌링턴|67|0|91|122|153|
|화이트 리버 정션|46|91|0|98|65|
|베닝턴|55|122|98|0|40|
|브래틀보로|75|153|65|40|0|

도시 간의 거리를 쉽게 찾을 수 있도록 딕셔너리의 딕셔너리를 사용한다.


```python
# tsp.py
from typing import Dict, List, Iterable, Tuple
from itertools import permutations

vt_distances: Dict[str, Dict[str, int]] = {
    "러틀랜드": {
        "벌링턴": 67,
        "화이트 리버 정션": 46,
        "베닝턴": 55,
        "브래틀보로": 75
    },
    "벌링턴": {
        "러틀랜드": 67,
        "화이트 리버 정션": 91,
        "베닝턴": 122,
        "브래틀보로": 153
    },
    "화이트 리버 정션": {
        "러틀랜드": 46,
        "벌링턴": 91,
        "베닝턴": 98,
        "브래틀보로": 65
    },
    "베닝턴": {
        "러틀랜드": 55,
        "벌링턴": 122,
        "화이트 리버 정션": 98,
        "브래틀보로": 40
    },
    "브래틀보로": {
        "러틀랜드": 75,
        "벌링턴": 153,
        "화이트 리버 정션": 65,
        "베닝턴": 40
    }
}
```

도시들의 모든 순열을 찾기 위해 백트래킹을 사용할 수 있다. 각 도시를 스왑하여 추가 순열 경로를 생성한 후 다른 경로를 찾기 위해 순열 경로의 각 도시를 스왑한다. 이때 스왑이 수행되기 전의 상태로 백트래킹(역추적)할 수 있다.  
  
파이썬 표준 라이브러리의 **itertools** 모듈에 `permutation()` 함수는 순열을 생성해준다. 예제에서 5개의 도시를 방문해야 하기 때문에 시간복잡도는 $5!(120 = 5 \times 4 \times 3 \times 2 \times 1)$ 이다.  
  
<hr>

### 브루트포스 탐색
외판원이 방문할 마지막 도시는 맨 처음 출발한 도시여야함으로, 리스트 컴프리헨션을 통해 순열의 첫 번째 도시를 추가한다.
모든 순열의 도시 경로의 거리를 합하여 최단 경로의 도시를 나열하고, 총 거리를 출력한다.


```python
# tsp.py
...

vt_cities: Iterable[str] = vt_distances.keys()
city_permutations: Iterable[Tuple[str, ...]] = permutations(vt_cities)
tsp_paths: List[Tuple[str, ...]] = [c + (c[0],) for c in city_permutations]

if __name__ == "__main__":
    best_path: Tuple[str, ...]
    min_distance: int = 999999999 # 높은 숫자로 설정
    for path in tsp_paths:
        distance: int = 0
        last: str = path[0]
        for next in path[1:]:
            distance += vt_distances[last][next]
            last = next
        if distance < min_distance:
            min_distance = distance
            best_path = path
    print(f"최단 경로 {best_path} , {min_distance} 마일입니다.")
```

    최단 경로 ('러틀랜드', '벌링턴', '화이트 리버 정션', '브래틀보로', '베닝턴', '러틀랜드') , 318 마일입니다.
    

### 더 좋은 방법
외판원 문제에 대한 완벽한 해결책은 아직 없으며, 앞서 소개한 브루트 포스 방식은 많은 시간이 소요된다. 시간복잡도는 $O(n!)$ 로 실생활에서 쓰기에 어렵다. 많은 도시가 있는 경우 대부분의 알고리즘은 최단 경로에 가까운 근사치를 반환한다. 알고리즘은 [동적 계획법]()과 [유전 알고리즘](https://moongni.github.io/algorithm/Genetic-Algorithm/)이 있으며, 최단 경로를 찾으려고 노력한다.  

## 전화번호 니모닉
스마트폰의 전화 앱에는 숫자 버튼에 문자가 포함되어있다. 이러한 문자가 있는 이유는 전화번호를 기억하기 쉬운 **니모닉**으로 제공하기 위함이다. 예를 들어 1-800-MY-APPLE은 전화번호 1-800-69-27753에 해당한다.  
<center><img src="/assets/images/posting/mnemonic.jpg" width="20%" height="20%"></center>

전화번호에서 문자의 모든 순열을 생성한 다음 사전을 통해 순열에 포함된 단어를 찾는다. 전화번호에서 각 숫자와 잋리하는 문자를 보면서 계속 문자를 결합해 나간다. 일종의 **cartesian product**(카티션 곱, 집합 A와 집합 B를 곱한 집합)이다.  
파이썬 표준 라이브러리의 **itertools** 모듈에 `product()` 함수를 이용한다.  


```python
#phone_number_mnemonics.py
from typing import Dict, Tuple, Iterable, List
from itertools import product

phone_mapping: Dict[str, Tuple[str, ...]] = {
    "1" : ("1", ),
    "2" : ("a", "b", "c"),
    "3" : ("d", "e", "f"),
    "4" : ("g", "h", "i"),
    "5" : ("j", "k", "l"),
    "6" : ("m", "n", "o"),
    "7" : ("p", "q", "r", "s"),
    "8" : ("t", "u", "v"),
    "9" : ("w", "x", "y", "z"),
    "0" : ("0", )
}
    
def possible_mnemonics(phone_number: str) -> Iterable[Tuple[str, ...]]:
    letter_tuples: List[Tuple[str, ...]] = []
    for digit in phone_number:
        letter_tuples.append(phone_mapping.get(digit, (digit, )))
    return product(*letter_tuples)
```

`possible_mnemonics` 함수는 전화번호의 각 숫자에서 가능한 모든 문자를 니모닉 리스트로 결합한다. 전화번호의 각 숫자에 대한 잠재적 문자의 튜플 리스트를 작성한 후 `product()` 함수를 사용하여 문자를 결합한다. 언팩연산자 `*` 를 사용하여 `product()` 함수의 인자로 사용한다.  


```python
# phone_number_mnemonics.py
if __name__ == '__main__':
    phone_number: str = input('전화번호를 입력해주세요')
    print('가능한 니모닉 목록: ')
    for mnemonic in possible_mnemonics(phone_number):
        print("".join(mnemonic))
```

    전화번호를 입력해주세요1440787
    가능한 니모닉 목록: 
    1gg0ptp
    1gg0ptq
    
    ...
    
    1gh0str
    1gh0sts
    1gh0sup

    ...

    1ii0svr
    1ii0svs
    

> 전화번호 1440787 은 니모닉 문자로 1gh0sts 로 사용할 수 있다.

## 연습문제
### 1.
### 2.
### 3.