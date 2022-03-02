---
layout: single
title: "[Algorithm] 유전 알고리즘"
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
    <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#%EC%9C%A0%EC%A0%84-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98">유전 알고리즘</a>
    <ul>
        <li>
            <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#51-%EC%8B%9C%EB%AE%AC%EB%A0%88%EC%9D%B4%EC%85%98">시뮬레이션</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#52-%EC%A0%9C%EB%84%A4%EB%A6%AD-%EC%9C%A0%EC%A0%84-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98">제네릭 유전 알고리즘</a>
        </li>
    </ul>
    <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#%ED%99%9C%EC%9A%A9">활용 예제</a>
    <ul>
        <li>
            <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#53-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B0%A9%EC%A0%95%EC%8B%9D">간단한 방정식</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#54-send--more--money">SEND+MORE=MONEY</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#55-%EC%B5%9C%EC%A0%81%ED%99%94-%EB%A6%AC%EC%8A%A4%ED%8A%B8-%EC%95%95%EC%B6%95">최적화 리스트 압축</a>
        </li>
    </ul>
    <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#%EC%97%B0%EC%8A%B5%EB%AC%B8%EC%A0%9C">연습문제</a>
    <ul>
        <li>
            <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#1-%EC%B2%B4%EA%B0%90%ED%99%95%EC%9C%A8%EC%9D%84-%EA%B8%B0%EB%B0%98%EC%9C%BC%EB%A1%9C-%EB%95%8C%EB%A1%9C%EB%8A%94-%EB%91%90-%EB%B2%88%EC%A7%B8-%ED%98%B9%EC%9D%80-%EC%84%B8-%EB%B2%88%EC%A7%B8%EB%A1%9C-%EA%B0%80%EC%9E%A5-%EC%A2%8B%EC%9D%80-%EC%97%BC%EC%83%89%EC%B2%B4%EB%A5%BC-%EC%84%A0%ED%83%9D%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8A%94-%EA%B3%A0%EA%B8%89-%ED%86%A0%EB%84%88%EB%A8%BC%ED%8A%B8-%EC%84%A0%ED%83%9D-%EC%9C%A0%ED%98%95%EC%9D%84-geneticalgorithm-%ED%81%B4%EB%9E%98%EC%8A%A4%EC%97%90-%EC%B6%94%EA%B0%80%ED%95%98%EB%9D%BC">1 번</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#2-%EC%A0%9C%EC%95%BD-%EC%B6%A9%EC%A1%B1-%EB%AC%B8%EC%A0%9C-%ED%94%84%EB%A0%88%EC%9E%84%EC%9B%8C%ED%81%AC%EC%97%90-%EC%9C%A0%EC%A0%84-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC-%EC%9E%84%EC%9D%98%EC%9D%98-%EC%A0%9C%EC%95%BD-%EC%B6%A9%EC%A1%B1-%EB%AC%B8%EC%A0%9C%EB%A5%BC-%ED%95%B4%EA%B2%B0%ED%95%98%EB%8A%94-%EC%83%88%EB%A1%9C%EC%9A%B4-%EB%A9%94%EC%84%9C%EB%93%9C%EB%A5%BC-%EC%B6%94%EA%B0%80%ED%95%98%EB%9D%BC-%EC%A0%81%ED%95%A9%EB%8F%84%EC%9D%98-%EA%B0%80%EB%8A%A5%ED%95%9C-%EC%B8%A1%EC%A0%95%EC%9D%80-%EC%97%BC%EC%83%89%EC%B2%B4%EC%97%90-%EC%9D%98%ED%95%B4-%ED%95%B4%EA%B2%B0%EB%90%98%EB%8A%94-%EC%A0%9C%EC%95%BD%EC%A1%B0%EA%B1%B4%EC%9D%98-%EC%88%98%EB%8B%A4">2 번</a>
        </li>
        <li>
            <a href="https://moongni.github.io/algorithm/Genetic-Algorithm/#3-chromosome-%ED%81%B4%EB%9E%98%EC%8A%A4%EB%A5%BC-%EC%83%81%EC%86%8D%EB%B0%9B%EB%8A%94-bitstring-%ED%81%B4%EB%9E%98%EC%8A%A4%EB%A5%BC-%EC%83%9D%EC%84%B1%ED%95%98%EA%B3%A0-53-%EA%B0%84%EB%8B%A8%ED%95%9C-%EB%B0%A9%EC%A0%95%EC%8B%9D-%EB%AC%B8%EC%A0%9C%EB%A5%BC-%ED%95%B4%EA%B2%B0%ED%95%98%EB%9D%BC">3 번</a>
        </li>
    </ul>
</div>

## 유전 알고리즘
유전 알고리즘은 일반적으로 쉬운 해결책이 없으며 복잡한 문제를 위한 알고리즘이다.  
예시로 단백질-리간드 결합과 약물 설계이다. 약물을 전달하기 위해 수용체에 결합할 분자를 설계한다.  
  
### 5.1 시뮬레이션
유전 알고리즘은 **염색체**로 알려진 개체의 모집단(각자의 특성을 나타내는 유전자로 구성된 염색체들)은 어떤 문제를 해결하기 위해 경쟁하고 있다고 할 때, 염색체가 문제를 얼마나 잘 해결하는지는 **적합도 함수**에 의해 정의된다.  
  
세대를 거치며 문제를 해결할 수 있는 염색체는 **선택**될 가능성이 크다. 세대마다 두 개의 염색체가 유전자를 합칠 가능성(**크로스오버**), 세대마다 염색체의 유전자가 무작위로 **변이**될 가능성이 있다.  
  
모집단의 일부 개체가 적합도 함수에서 지정된 임계값을 초과하거나 알고리즘이 지정한 세대의 최고값을 통과한다면 가장 적합한 개체가 반환된다.  
  
**유전 알고리즘은 모든 문제의 해결책이 아니다. 위에 시뮬레이션에서도 선택, 크로스오버, 변이 등의 확률에 의존한다. 그러나 빠른 결정론적 알고리즘이 존재하지 않는 문제에서 유전 알고리즘은 좋은 선택이될 수 있다.**

<hr>

### 5.2 제네릭 유전 알고리즘
<div class="notice--info">
<h4>아래 4개의 기능을 가진 <b>Chromosome</b> 추상 클래스를 정의한다.</h4>

<ul>
    <li>
        자체 적합도를 결정한다.
    </li>
    <li>
        (첫 세대에서 사용하기 위해)무작위로 선택된 유전자로 인스턴스를 생성한다.
    </li>
    <li>
        다른 염색체와 혼합되는 크로스오버를 구현한다.
    </li>
    <li>
        돌연변이를 구현한다. 즉 자체로 작고 무작위적인 변화를 만든다.
    </li>
</ul>
</div>


```python
#chromosome.py
from __future__ import annotations
from typing import TypeVar, Tuple, Type
from abc import ABC, abstractmethod

T = TypeVar('T', bound = 'Chromosome') # 자신을 반환하기 위해 사용

# 모든 염색체의 베이스 클래스, 모든 메서드는 오버라이드 된다.
class Chromosome(ABC):
    @abstractmethod
    def fitness(self) -> float:
        ...
    
    @classmethod
    @abstractmethod
    def random_instance(cls: Type[T]) -> T:
        ...
        
    @abstractmethod
    def crossover(self: T, other: T) -> Tuple[T, T]:
        ...
    
    @abstractmethod
    def mutate(self) -> None:
        ...
```

<hr>

특정 애플리케이션을 위해 상속 가능한 제네릭 클래스로 염색체를 조작하는 알고리즘을 구현한다.  
  
<div class="notice--info">
    <h4>유전자 알고리즘의 단계</h4>
        <ol start="1">
            <li>
                유전 알고리즘 1세대에 대한 무작위 염색체 초기 모집단 생성
            </li>
            <li>
                세대 집단에서 각 염색체의 적합도를 측정한다. 임곗값을 초과하는 염색체가 존재하면 반환하고 종료한다.
            </li>
            <li>
                개체의 재생산을 위해 가장 높은 적합도를 가진 확률이 높은 개체를 선택한다.
            </li>
            <li>
                다음 세대 집단 생성하기 위해 일정 확률로 크로스오버(결합) 한다.
            </li>
            <li>
                낮은 확률로 일부 염색체를 돌연변이시킨다. 마지막 세대의 집단을 대체한다.
            </li>
            <li>
                알고리즘으로 지정된 세대의 최댓값을 도달하지 못한 경우 2단계로 돌아간다. 최댓값에 도달시 적합도가 가장 높은 염색체를 반환한다.  
            </li>
        </ol>
</div>

```python
#genetic_algorithm.py
from __future__ import annotations
from typing import TypeVar, Generic, List, Tuple, Callable
from enum import Enum
from random import choices, random
from heapq import nlargest
from statistics import mean
from chromosome import Chromosome

C = TypeVar('C', bound=Chromosome) # 염색체 타입

class GeneticAlgorithm(Generic[C]):
    SelectionType = Enum("SelectionType", "ROULETTE TOURNAMENT") # 알고리즘 내부에 사용되는 방식: 룰렛, 토너먼트
    
    def __init__(self, initial_population: List[C], threshold: float, max_generations: int = 100,
                mutation_chance: float = 0.01, crossover_chance: float = 0.7,
                selection_type: SelectionType = SelectionType.TOURNAMENT) -> None:
        self._population: List[C] = initial_population
        self._threshold: float = threshold                      # 적합도 수준 임곗값
        self._max_generations: int = max_generations            # 최대 세대 수 default: 100
        self._mutation_chance: float = mutation_chance          # 돌연변이 확률 default: 0.01
        self._crossover_chance: float = crossover_chance        # 결합 확률 default: 0.7
        self._selection_type: GeneticAlgorithm.SelectionType = selection_type
        self._fitness_key: Callable = type(self._population[0]).fitness
            
    # 두 부모를 선택하기 위해 룰렛휠(활률 분포)을 사용한다.
    # 메모: 음수 적합도와 작동하지 않는다.
    def _pick_roulette(self, wheel: List[float]) -> Tuple[C, C]:
        return tuple(choices(self._population, weights=wheel, k=2)) # wheel: 각 염색체의 적합도
    
    # 무작위로 num_participants만큼 추출한 후 적합도가 가장 높은 두 염색체를 선택한다.
    def _pick_tournament(self, num_participants: int) -> Tuple[C, C]:
        participants: List[C] = choices(self._population, k=num_participants)
        return tuple(nlargest(2, participants, key=self._fitness_key)) 
    
    # 집단을 새로우 세대로 교체한다.
    def _reproduce_and_replace(self) -> None:
        new_population: List[C] = []
        # 새로운 세대가 채워질 때까지 반복한다.
        while len(new_population) < len(self._population):
            # parents 중 두 부모를 선택한다.
            if self._selection_type == GeneticAlgorithm.SelectionType.ROULETTE:
                parents: Tuple[C, C] = self._pick_roulette([x.fitness() for x in self._population])
            else:
                parents: Tuple[C, C] = self._pick_tournament(len(self._population) // 2)
            # 두 부모의 크로스 오버
            if random() < self._crossover_chance: # random < 0.7
                new_population.extend(parents[0].crossover(parents[1]))
            else:
                new_population.extend(parents)
        # 집단의 수가 홀수일 경우 2개씩 추가되기 때문에
        # 이전 집단보다 하나 더 많을 수 있다.
        if len(new_population) > len(self._population):
            new_population.pop()
        # 새로운 세대로 참조를 변경한다.
        self._population = new_population         

    # _mutation_chance 확률로 각 개별 염색체를 돌연변이한다.
    def _mutate(self) -> None:
        for individual in self._population:
            # 돌연변이 생성
            if random() < self._mutation_chance: # random < 0.01
                individual.mutate()
                
    def run(self) -> C:
        best: C = max(self._population, key=self._fitness_key)
        for generation in range(self._max_generations):
            # 임곗값을 초과하면 반환한다.
            if best.fitness() >= self._threshold:
                return best
            print(f"세대 {generation} 최상 {best.fitness()} 평균 {mean(map(self._fitness_key, self._population))}")
            # 다음 세대로 이동하기 위한 선택, 크로스오버, 돌연변이
            self._reproduce_and_replace()
            self._mutate()
                  
            highest: C = max(self._population, key=self._fitness_key)
            if highest.fitness() > best.fitness():
                  best = highest # 새로운 최상의 개체를 발견
                  
        return best # _max_generation에서 발견한 최상의 개체를 반환한다.
```

```python
self._fitness_key: Callable = type(self._population[0]).fitness
```
- `_fitness_key` 변수는 GeneticAlgorithm 클래스를 통해 염색체의 적합도를 계산하는 메서드의 참조이다. GeneticAlgorithm 클래스는 Chromosome 클래스의 서브클래스와 실행되므로 `_fitness_key` 변수는 서브클래스에 따라 다를 수 있다. 그 값을 찾기 위해 `type()` 함수를 사용하여 적합도를 찾는 염색체의 특정 서브클래스를 참조한다.

<div class="notice--info">

<h4>_reproduce_and_replace() 단계</h4>
<ol>
<li>
parents의 두 개의 염색체를 룰렛휠과 토너먼트 중 하나의 방식으로 재생산을 위해 선택된다. 토너먼트 선택에서 <b>len(self._population) // 2</b> 로 토너먼트를 하지만 구성 옵션을 변경할 수 있다.
</li>
<li>
parents를 결합하여 두 개의 염색체를 생성하는 결합확률에 따라 <b>new_population</b>에 자식 또는 부모를 추가한다.
</li>
<li>
new_population에 self._population의 크기만큼 들어가지 않았다면 1단계로 돌아간다. 그렇지 않으면 새 집단으로 대체한다
</li>
</ol>
</div>

## 활용
### 5.3 간단한 방정식

**1. 방정식 $6x - x^2 + 4y - y^2$이 최댓값이 되는 $x$와 $y$는 무엇일까?**  
  
미적분을 사용하여 편미분을 취하고 각각 0으로 설정하면 솔루션을 찾을 수 있다.  
  
$x = (6 - 2x = 0)$ , $y = (4 - 2y = 0)$  >>  $x = 3, y = 2$  
  
유전 알고리즘을 사용해 같은 결과에 도달할 수 있는가


```python
#simple_equation.py
from __future__ import annotations
from typing import Tuple, List
from Chromosome import Chromosome
from genetic_algorithm import GeneticAlgorithm
from random import randrange, random
from copy import deepcopy

class SimpleEquation(Chromosome):
    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y
    
    def fitness(self) -> float: # 6x - x^2 + 4y - y^2
        return 6 * self.x - self.x ** 2 + 4 * self.y - self.y ** 2
    
    # 첫 세대로 x와 y의 값이 0 ~ 100 사이의 임의의 정수로 설정한다.
    @classmethod
    def random_instance(cls) -> SimpleEquation:
        return SimpleEquation(randrange(100), randrange(100))
    
    def crossover(self, other: SimpleEquation) -> Tuple[SimpleEquation, SimpleEquation]:
        child1: SimpleEquation = deepcopy(self)
        child2: SimpleEquation = deepcopy(other)
        # 두 인스턴스의 y값을 바꿔 두 자식을 만든다.
        child1.y, child2.y = other.y, self.y

        return child1, child2
    
    def mutate(self) -> None:
        # x를 돌연변이한다.
        if random() > 0.5: 
            if random() > 0.5:
                self.x += 1
            else:
                self.x -= 1
        # y를 돌연변이한다.
        else: 
            if random() > 0.5:
                self.y += 1
            else:
                self.y -= 1
    
    def __str__(self) -> str:
        return f"X: {self.x} Y: {self.y} 적합도: {self.fitness()}"
```


```python
# 실행
initial_population: List[SimpleEquation] = [SimpleEquation.random_instance() for _ in range(20)]
ga: GeneticAlgorithm[SimpleEquation] = GeneticAlgorithm(initial_population = initial_population,
                                                        threshold = 13.0, max_generations = 100,
                                                       mutation_chance = 0.1, crossover_chance=0.7)
result: SimpleEquation = ga.run()
print(result)
```

    세대 0 최상 12 평균 -5643.2
    세대 1 최상 12 평균 -449.3
    세대 2 최상 12 평균 -21.95
    세대 3 최상 12 평균 8.95
    X: 3 Y: 2 적합도: 13
    

> **유전 알고리즘은 다른 해결 방법보다 더 많은 계산이 필요하다.** 따라서 합리적인 신간 내에 최적의 솔루션을 찾지 못할 수도 있기 때문에 단순 최대화 문제에 유전 알고리즘을 잘 적용할 수 없다.

<hr>

### 5.4 SEND + MORE = MONEY
[제약충족문제](https://moongni.github.io/algorithm/Constraint-Satisfaction-Problems/#35-sendmoremoney)에서 봤던 SEND+MORE=MONEY문제를 유전 알고리즘을 통해 합리적인 시간 내에 해결할 수 있다는 것을 보여준다.

문제 설명 : “S”,”E”,”N”,”D”,”M”,”O”,”R”,”Y” 8 개의 문자를 (0 ~ 9) 까지의 한 자리의 숫자를 중복없이 넣어 SEND+MORE = MONEY 를 만족해야 한다.

가능한 10자리(0 ~ 9)를 리스트 인덱스를 이용하여 표현한다.  
`list[4] = "E"` 식으로 표현하며 리스트에 두 자리 공백은 문자가 없음을 의미한다.


```python
#send_more_money2.py
from __future__ import annotations
from typing import Tuple, List
from Chromosome import Chromosome
from genetic_algorithm import GeneticAlgorithm
from random import shuffle, sample
from copy import deepcopy

class SendMoreMoney2(Chromosome):
    def __init__(self, letters: List[str]) -> None:
        self.letters: List[str] = letters
            
    def fitness(self) -> float:
        s: int = self.letters.index("S")
        e: int = self.letters.index("E")
        n: int = self.letters.index("N")
        d: int = self.letters.index("D")
        m: int = self.letters.index("M")
        o: int = self.letters.index("O")
        r: int = self.letters.index("R")
        y: int = self.letters.index("Y")
        send: int = s * 1000 + e * 100 + n * 10 + d
        more: int = m * 1000 + o * 100 + r * 10 + e
        money: int = m * 10000 + o * 1000 + n * 100 + e * 10 + y
        difference: int = abs(money - (send + more))
        
        return 1 / (difference + 1)
    
    # 첫 세대로 8개의 문자를 섞는다.
    @classmethod
    def random_instance(cls) -> SendMoreMoney2:
        letters = ["S","E","N","D","M","O","R","Y"," "," "]
        shuffle(letters)
        return SendMoreMoney2(letters)
    
    # 두 부모의 각각 두 문자의 위치 섞는다.
    def crossover(self, other: SendMoreMoney2) -> Tuple(SendMoreMoney2, SendMoreMoney2):
        child1: SendMoreMoney2 = deepcopy(self)
        child2: SendMoreMoney2 = deepcopy(other)
        idx1, idx2 = sample(range(len(self.letters)), k = 2)
        l1, l2 = child1.letters[idx1], child2.letters[idx2]
        child1.letters[child1.letters.index(l2)], child1.letters[idx2] = child1.letters[idx2], l2
        child2.letters[child2.letters.index(l1)], child2.letters[idx1] = child2.letters[idx1], l1
        return child1, child2
    
    def mutate(self) -> None: # 두 문자의 위치를 스왑한다.
        idx1, idx2 = sample(range(len(self.letters)), k = 2)
        self.letters[idx1], self.letters[idx2] = self.letters[idx2], self.letters[idx1]
        
    def __str__(self) -> str:
        s: int = self.letters.index("S")
        e: int = self.letters.index("E")
        n: int = self.letters.index("N")
        d: int = self.letters.index("D")
        m: int = self.letters.index("M")
        o: int = self.letters.index("O")
        r: int = self.letters.index("R")
        y: int = self.letters.index("Y")
        send: int = s * 1000 + e * 100 + n * 10 + d
        more: int = m * 1000 + o * 100 + r * 10 + e
        money: int = m * 10000 + o * 1000 + n * 100 + e * 10 + y
        difference: int = abs(money - (send + more))
        return f"{send} + {more} = {money} 차이: {difference}"
```


```python
initial_population: List[SendMoreMoney2] = [SendMoreMoney2.random_instance() for _ in range(1000)]
ga: GeneticAlgorithm[SendMoreMoney2] = GeneticAlgorithm(initial_population = initial_population,
                                                       threshold = 1.0, max_generations = 1000,
                                                       mutation_chance = 0.2, crossover_chance = 0.7,
                                                       selection_type = GeneticAlgorithm.SelectionType.ROULETTE)
result: SendMoreMoney2 = ga.run()
print(result)
```

    세대 0 최상 0.00909090909090909 평균 0.0001077711615040916
    세대 1 최상 0.09090909090909091 평균 0.0013404556154218365
    세대 2 최상 0.14285714285714285 평균 0.010707204374158672
    세대 3 최상 0.5 평균 0.03482710497875595
    3821 + 468 = 4289 차이: 0
    

> **결과 SEND = 3821, MORE = 468, MONEY = 4289 는 M = 0 으로 0이 무시되어 MORE = 0468, MONEY = 04289 이다.**

**fitness() 적합도 산출방법**

difference|difference + 1|fitness(1 / (difference + 1))
:---:|:---:|:---:
0|1|1
1|2|0.5
2|3|0.25
3|4|0.125


**CAUTION**: 1을 적합도로 나누는 것은 최소화 문제를 최대화 문제로 변환하는 간단한 방법이지만 **편향**이 있어서 절대로 안전한 방법은 아니다.  
1을 정수의 균일분포로 나누는 경우 0에 가까운 숫자가 많이 생길 수 있다. **>>** 일반적인 마이크로프로세서가 부동소수점수를 해석하는 미묘한 방식에 따라 예기치 않은 결과가 발생할 수 있다.
{: .notice--warning}

<hr>

### 5.5 최적화 리스트 압축
리스트로 구성된 정보를 압축하려고 할때, 어떤 순서가 모든 항목이 손상되지 않으면서 압축비율을 높일 수 있는가의 문제이다.  
  
파이썬 표준 라이브러리 `zlib` 모듈의 `compress()` 함수를 사용하여 12개의 이름 리스트를 유전 알고리즘을 이용하여 최적화 된 순서를 구한다.


```python
# list_compression.py
from __future__ import annotations
from typing import Tuple, List, Any
from chromosome import Chromosome
from genetic_algorithm import GeneticAlgorithm
from random import shuffle, sample
from copy import deepcopy
from zlib import compress
from sys import getsizeof
from pickle import dumps

# 이 순서로 압축 시 165 bytes
PEOPLE: List[str] = ["Michael", "Sarah", "Joshua",
                     "Narine", "David", "Sajid", 
                     "Melanie", "Daniel", "Wei", 
                     "Dean", "Brian", "Murat", "Lisa"]

class ListCompression(Chromosome):
    def __init__(self, lst: List[Any]) -> None:
        self.lst: List[Any] = lst

    @property
    def bytes_compressed(self) -> int:
        return getsizeof(compress(dumps(self.lst)))

    def fitness(self) -> float:
        return 1 / self.bytes_compressed
    
    # 첫 세대로 List 항목의 순서를 섞는다.
    @classmethod
    def random_instance(cls) -> ListCompression:
        mylst: List[str] = deepcopy(PEOPLE)
        shuffle(mylst)
        return ListCompression(mylst)

    def crossover(self, other: ListCompression) -> Tuple[ListCompression, ListCompression]:
        child1: ListCompression = deepcopy(self)
        child2: ListCompression = deepcopy(other)
        idx1, idx2 = sample(range(len(self.lst)), k=2)
        l1, l2 = child1.lst[idx1], child2.lst[idx2]
        child1.lst[child1.lst.index(l2)], child1.lst[idx2] = child1.lst[idx2], l2
        child2.lst[child2.lst.index(l1)], child2.lst[idx1] = child2.lst[idx1], l1
        return child1, child2

    def mutate(self) -> None: # 두 위치를 섞는다.
        idx1, idx2 = sample(range(len(self.lst)), k=2)
        self.lst[idx1], self.lst[idx2] = self.lst[idx2], self.lst[idx1]

    def __str__(self) -> str:
        return f"순서: {self.lst} Bytes: {self.bytes_compressed}"
```


```python
initial_population: List[ListCompression] = [ListCompression.random_instance() for _ in range(100)]
ga: GeneticAlgorithm[ListCompression] = GeneticAlgorithm(initial_population=initial_population, threshold=1.0, max_generations = 100, mutation_chance = 0.2, crossover_chance = 0.7, selection_type=GeneticAlgorithm.SelectionType.TOURNAMENT)
result: ListCompression = ga.run()
print(result)
```

    세대 0 최상 0.006172839506172839 평균 0.0060413670966407325
    세대 1 최상 0.006172839506172839 평균 0.006098932383418525
    세대 2 최상 0.006172839506172839 평균 0.006141294575116651
    세대 3 최상 0.006172839506172839 평균 0.006160872115193966
    
    ...

    세대 97 최상 0.00625 평균 0.0062263376506839245
    세대 98 최상 0.00625 평균 0.006218719813900178
    세대 99 최상 0.00625 평균 0.006229742205109266
    순서: ['Michael', 'Melanie', 'Narine', 'Joshua', 'Wei', 'Daniel', 'Dean', 'David', 'Brian', 'Murat', 'Lisa', 'Sajid', 'Sarah'] Bytes: 160
    

원래 순서보다 5바이트 절약할 수 있다. 하지만 최적의 순서인지는 알 수 없다. 12개 항목에 대한 리스트의 경우 479,001,600(12!)개의 순서가 있다.  
  
**궁극적으로 솔루션이 최적인지 여부는 모르지만 최적의 솔루션을 찾으려고 하는 유전 알고리즘을 사용할 수 있다.**

## 유전 알고리즘 정리
유전 알고리즘은 만병통치약이 아니며 실제 문제에서 대부분 적합하지 않다.  
크로스오버와 돌연변이 등의 확률적 특성으로 인해 실행시간을 예측할 수 없다. 또한 최적의 솔루션을 찾았는지 확실하게 알 수 없다.  
  
유전 알고리즘은 더 나은 솔루션이 존재하지 않는다고 확신할 때만 선택해야 한다.  
  
> 유전 알고리즘은 최적의 솔루션을 요구하지 않고 **충분히 좋은** 솔루션을 요구하는 상황에서 잘 활용될 수 있다. 

## 연습문제
### 1. 체감확율을 기반으로 때로는 두 번째 혹은 세 번째로 가장 좋은 염색체를 선택할 수 있는 고급 토너먼트 선택 유형을 GeneticAlgorithm 클래스에 추가하라

```python
#genetic_algorithm.py
...

class GeneticAlgorithm(Generic[C]):
    # 알고리즘 내부에 사용되는 방식: 룰렛, 토너먼트, 고급 토너먼트
    SelectionType = Enum("SelectionType", "ROULETTE TOURNAMENT ADVANCEDTOURNAMENT")

    ...
    
    # 적합도 기준으로 상위 3개 중 2개를 무작위로 뽑는다.
    def _pick_adtournament(self) -> Tuple[C, C]:
        participants: List[C] = list(nlargest(3, self._population, key=self._fitness_key))
        return tuple(choices(participants, k=2))
    
    def _reproduce_and_replace(self) -> None:
        new_population: List[C] = []
        while len(new_population) < len(self._population):
            if self._selection_type == GeneticAlgorithm.SelectionType.ROULETTE:
                parents: Tuple[C, C] = self._pick_roulette([x.fitness() for x in self._population])

            # ADVANCEDTOURNAMENT 방식으로 부모를 선정한다.
            elif self._selection_type == GeneticAlgorithm.SelectionType.ADVANCEDTOURNAMENT:
                parents: Tuple[C, C] = self._pick_adtournament()
            
            else:
                parents: Tuple[C, C] = self._pick_tournament(len(self._population) // 2)
            if random() < self._crossover_chance: # random < 0.7
                new_population.extend(parents[0].crossover(parents[1]))
            else:
                new_population.extend(parents)
        if len(new_population) > len(self._population):
            new_population.pop()

        self._population = new_population
        
    ...
```

```python
# 실행
initial_population: List[SimpleEquation] = [SimpleEquation.random_instance() for _ in range(20)]
ga: GeneticAlgorithm[SimpleEquation] = GeneticAlgorithm(initial_population = initial_population,
                                                        threshold = 13.0, max_generations = 100,
                                                       mutation_chance = 0.1, crossover_chance=0.7,
                                                       selection_type = GeneticAlgorithm.SelectionType.ADVANCEDTOURNAMENT)
result: SimpleEquation = ga.run()
print(result)
```

    세대 0 최상 -16 평균 -6033.4
    세대 1 최상 -27 평균 -292.65
    세대 2 최상 -16 평균 -21.5
    세대 3 최상 -16 평균 -16.55
    세대 4 최상 -7 평균 -13.3
    세대 5 최상 0 평균 -6.55
    세대 6 최상 0 평균 -3.8
    세대 7 최상 5 평균 -0.3
    세대 8 최상 8 평균 3.15
    세대 9 최상 8 평균 7.25
    세대 10 최상 9 평균 6.65
    세대 11 최상 9 평균 8.2
    세대 12 최상 9 평균 8.8
    세대 13 최상 12 평균 9.3
    세대 14 최상 12 평균 11.05
    세대 15 최상 12 평균 11.95
    세대 16 최상 12 평균 12
    세대 17 최상 12 평균 11.65
    X: 3 Y: 2 적합도: 13
    

### 2. 제약 충족 문제 프레임워크에 유전 알고리즘을 사용하여 임의의 제약 충족 문제를 해결하는 새로운 메서드를 추가하라. 적합도의 가능한 측정은 염색체에 의해 해결되는 제약조건의 수다.

지도 색칠문제  
- 변수 : 호주의 7개 지역 (뉴사우스웨일스, 빅토리아, 퀸즐랜드, 사우스 오스트레일리아, 웨스턴 오스트레일리아, 태즈메이니아, 노던 준주)  
- 도메인 : 3가지 색상 (빨강, 파랑, 녹색)  
- 제약 : 인접한 두 지역을 같은 색상을 할당할 수 없다.  


```python
#MapColoringConstraint.py
from __future__ import annotations
from typing import Tuple, List
from Chromosome import Chromosome
from genetic_algorithm import GeneticAlgorithm
from random import randrange, random, sample
from copy import deepcopy

class MapColoringConstraint(Chromosome):
    def __init__(self, place: List[str], color: List[str]) -> None:
        self.place: List[str] = place
        self.color: List[str] = color
    
    def fitness(self) -> float:
        count = 0
        if self.color[self.place.index("웨스턴 오스트레일리아")] != self.color[self.place.index("노던 준주")]:
            count += 1
        if self.color[self.place.index("웨스턴 오스트레일리아")] != self.color[self.place.index("사우스 오스트레일리아")]:
            count += 1
        if self.color[self.place.index("사우스 오스트레일리아")] != self.color[self.place.index("노던 준주")]:
            count += 1
        if self.color[self.place.index("사우스 오스트레일리아")] != self.color[self.place.index("퀸즐랜드")]:
            count += 1
        if self.color[self.place.index("퀸즐랜드")] != self.color[self.place.index("노던 준주")]:
            count += 1
        if self.color[self.place.index("퀸즐랜드")] != self.color[self.place.index("뉴사우스웨일스")]:
            count += 1
        if self.color[self.place.index("뉴사우스웨일스")] != self.color[self.place.index("사우스 오스트레일리아")]:
            count += 1
        if self.color[self.place.index("빅토리아")] != self.color[self.place.index("사우스 오스트레일리아")]:
            count += 1
        if self.color[self.place.index("빅토리아")] != self.color[self.place.index("뉴사우스웨일스")]:
            count += 1
        if self.color[self.place.index("빅토리아")] != self.color[self.place.index("태즈메이니아")]:
            count += 1
       
        return (count / 10)
    
    @classmethod
    def random_instance(cls) -> MapColoringConstraint:
        place: List[str] = ["뉴사우스웨일스", "빅토리아", "퀸즐랜드", "사우스 오스트레일리아",
                        "웨스턴 오스트레일리아", "태즈메이니아", "노던 준주"]
        color: List[str] = ["빨강", "초록", "파랑"]
        return MapColoringConstraint(place, list((choices(color, k=1)) for _ in range(len(place))))
    
    
    def crossover(self, other: MapColoringConstraint) -> Tuple[MapColoringConstraint, MapColoringConstraint]:
        child1: MapColoringConstraint = deepcopy(self)
        child2: MapColoringConstraint = deepcopy(other)
        idx1, idx2 = sample(range(len(child1.place)), k = 2)
        child1.color[idx1] , child2.color[idx2] = child2.color[idx2] , child1.color[idx1]
        return child1, child2
    
    def mutate(self) -> None:
        if random() > 0.5:
            idx1 = randrange(len(self.place))
            self.color[idx1] = choices(["빨강","초록","파랑"], k = 1)
    
    def __str__(self) -> str:
        return f"place  {self.place} \n color  {self.color}"
```


```python
initial_population: List[MapColoringConstraint] = [MapColoringConstraint.random_instance() for _ in range(20)]
ga: GeneticAlgorithm[MapColoringConstraint] = GeneticAlgorithm(initial_population = initial_population,
                                                              threshold = 1, max_generations = 100,
                                                              mutation_chance = 0.2, crossover_chance=0.7)
result: MapColoringConstraint = ga.run()
print(result)
```

    세대 0 최상 0.9 평균 0.655
    세대 1 최상 0.9 평균 0.78
    세대 2 최상 0.9 평균 0.805
    place  ['뉴사우스웨일스', '빅토리아', '퀸즐랜드', '사우스 오스트레일리아', '웨스턴 오스트레일리아', '태즈메이니아', '노던 준주']
    color  [['빨강'], ['파랑'], ['파랑'], ['초록'], ['파랑'], ['빨강'], ['빨강']]
    
### 3. Chromosome 클래스를 상속받는 BitString 클래스를 생성하고, 5.3 '간단한 방정식' 문제를 해결하라

