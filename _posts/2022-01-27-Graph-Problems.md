---
layout: single
title: "[Algorithm] 그래프 문제"
categories: algorithm
tags: [python, algorithm]
classes: wide
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---
# 고전 알고리즘 인 파이썬 - 그래프 문제
## 그래프 문제
그래프는 문제를 연결된 노드 집합으로 구성하여 모델링하는 데 사용하는 추상적인 수학 구조물이다.  

> 각 노드는 `정점 vertex` 라 하며 노드간 연결을 `에지 edge` 라고 한다.



## 4.2 그래프 프레임워크
두 가지 유형의 그래프가 있다.
1. 가중치가 없는 그래프: 에지의 가중치를 고려하지 않음
2. 가중치 그래프: 에지에 가중치(숫자)를 부여한 것

객체지향프로그밍에서 클래스 상속을 통해 두가지 유형의 그래프를 다룰 수 있도록 로직을 작성한다.


```python
# edge.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Edge:
    u: int # 정점 u에서(from)
    v: int # 정점 v로(to)
    
    def reversed(self) -> Edge:
        return Edge(self.v, self.u)

    def __str__(self) -> str:
        return f"{self.u} -> {self.v}"
```

방향성이 없는 그래프를 다룰 예정이기 때문에 `reversed()`는 에지의 반대 방향으로 이동하는 `Edge`를 반환한다.  
> 방향성이 없는 그래프: 에지가 양방향인 그래프

NOTE: dataclass 모듈을 사용한다. `@dataclass` 데커레이터로 표시된 클래스는 자동적으로 `__init__()` 메서드를 생성한다. 타입 어노테이션으로 선언된 변수를 자동으로 인스턴스화한다.
{: .notice--public}


```python
# graph.py
from typing import TypeVar, Generic, List, Optional
from edge import Edge

V = TypeVar('V') # 그래프 정점 타입

class Graph(Generic[V]):
    def __init__(self, vertices: List[V] = []) -> None:
        self._vertices: List[V] = vertices
        self._edges: List[List[Edge]] = [[] for _ in vertices]
    
    @property
    def vertex_count(self) -> int:
        return len(self._vertices) # 정점의 수
    
    @property
    def edge_count(self) -> int:
        return sum(map(len, self._edges)) # 에지의 수
    
    # 그래프에 정점을 추가하고 인덱스를 반환한다.
    def add_vertex(self, vertex: V) -> int:
        self._vertices.append(vertex)
        self._edges.append([]) # 에지에 빈 리스트를 추가
        return self.vertex_count - 1 # 추가된 정점의 인덱스를 반환
    
    # 방향이 없는 그래프이므로 양방향으로 에지를 추가
    def add_edge(self, edge: Edge) -> None:
        self._edges[edge.u].append(edge)
        self._edges[edge.v].append(edge.reversed())
    
    # 정점 인덱스를 사용하여 에지를 추가(헬퍼 메서드)
    def add_edge_by_indices(self, u:int, v:int) -> None:
        edge: Edge = Edge(u, v)
        self.add_edge(edge)
    
    # 정점 인덱스를 참조하여 에지를 추가(헬퍼 메서드)
    def add_edge_by_vertices(self, first: V, second: V) -> None:
        u: int = self._vertices.index(first)
        v: int = self._vertices.index(second)
        self.add_edge_by_indices(u, v)
    
    # 특정 인덱스에서 정점을 찾음
    def vertex_at(self, index: int) -> V:
        return self._vertices[index]

    # 정점 인덱스를 찾음
    def index_of(self, vertex: V) -> int:
        return self._vertices.index(vertex)
    
    # 정점 인덱스에 연결된 이웃 정점을 찾음
    def neighbors_for_index(self, index: int) -> List[V]:
        return list(map(self.vertex_at, [edge.v for edge in self._edges[index]]))
    
    # 정점의 이웃 정점을 찾음(헬퍼 메서드)
    def neighbors_for_vertex(self, vertex: V) -> List[V]:
        return self.neighbors_for_index(self.index_of(vertex))
    
    # 정점 인덱스에 연결된 모든 에지를 반환
    def edges_for_index(self, index: int) -> List[Edge]:
        return self._edges[index]
    
    # 정점의 해당 에지를 반환(헬퍼 메서드)
    def edges_for_vertex(self, vertex: V) -> List[Edge]:
        return self.edges_for_index(self.index_of(vertex))
    
    # 출력
    def __str__(self):
        desc: str = ""
        for i in range(self.vertex_count):
            desc += f"{self.vertex_at(i)} -> {self.neighbors_for_index(i)}\n"
        return desc
```

리스트 변수 `_vertices` 에 각 정점을 저장하고 리스트의 정수 인덱스로 참조한다.  
> **다른 두 정점이 같은 이름을 가졌더라도 정수 인덱스로 참조할 수 있다**  

이 절의 예제에서 자료구조를 **인접 리스트**를 사용한다.   
인접 리스트에서 모든 정점에는 해당 정점이 연결된 모든 정점 리스트가 있다.  

정점에 대한 인덱스를 모르는 경우 `_vertiecs` 를 검색하여 찾아야하기 때문에, 인덱스와 V 타입에 대한 각각의 메서드가 존재한다. 

### 4.2.1 Edge와 Graph 클래스 사용
구현된 `Edge` , `Graph` 클래스를 사용하여 미국 도시를 잇는 네트워크를 만든다.


```python
# graph.py 계속
city_graph: Graph[str] = Graph(["Seattle", "San Francisco", "Los Angeles",
                                    "Riverside", "Phoenix", "Chicago", "Boston",
                                    "New York", "Atlanta", "Miami", "Dallas",
                                    "Houston", "Detroit", "Philadelphia",
                                    "Washington"])
city_graph.add_edge_by_vertices("Seattle", "Chicago")
city_graph.add_edge_by_vertices("Seattle", "San Francisco")
city_graph.add_edge_by_vertices("San Francisco", "Riverside")
city_graph.add_edge_by_vertices("San Francisco", "Los Angeles")
city_graph.add_edge_by_vertices("Los Angeles", "Riverside")
city_graph.add_edge_by_vertices("Los Angeles", "Phoenix")
city_graph.add_edge_by_vertices("Riverside", "Phoenix")
city_graph.add_edge_by_vertices("Riverside", "Chicago")
city_graph.add_edge_by_vertices("Phoenix", "Dallas")
city_graph.add_edge_by_vertices("Phoenix", "Houston")
city_graph.add_edge_by_vertices("Dallas", "Chicago")
city_graph.add_edge_by_vertices("Dallas", "Atlanta")
city_graph.add_edge_by_vertices("Dallas", "Houston")
city_graph.add_edge_by_vertices("Houston", "Atlanta")
city_graph.add_edge_by_vertices("Houston", "Miami")
city_graph.add_edge_by_vertices("Atlanta", "Chicago")
city_graph.add_edge_by_vertices("Atlanta", "Washington")
city_graph.add_edge_by_vertices("Atlanta", "Miami")
city_graph.add_edge_by_vertices("Miami", "Washington")
city_graph.add_edge_by_vertices("Chicago", "Detroit")
city_graph.add_edge_by_vertices("Detroit", "Boston")
city_graph.add_edge_by_vertices("Detroit", "Washington")
city_graph.add_edge_by_vertices("Detroit", "New York")
city_graph.add_edge_by_vertices("Boston", "New York")
city_graph.add_edge_by_vertices("New York", "Philadelphia")
city_graph.add_edge_by_vertices("Philadelphia", "Washington")
print(city_graph)
```

    Seattle -> ['Chicago', 'San Francisco']
    San Francisco -> ['Seattle', 'Riverside', 'Los Angeles']
    Los Angeles -> ['San Francisco', 'Riverside', 'Phoenix']
    Riverside -> ['San Francisco', 'Los Angeles', 'Phoenix', 'Chicago']
    Phoenix -> ['Los Angeles', 'Riverside', 'Dallas', 'Houston']
    Chicago -> ['Seattle', 'Riverside', 'Dallas', 'Atlanta', 'Detroit']
    Boston -> ['Detroit', 'New York']
    New York -> ['Detroit', 'Boston', 'Philadelphia']
    Atlanta -> ['Dallas', 'Houston', 'Chicago', 'Washington', 'Miami']
    Miami -> ['Houston', 'Atlanta', 'Washington']
    Dallas -> ['Phoenix', 'Chicago', 'Atlanta', 'Houston']
    Houston -> ['Phoenix', 'Dallas', 'Atlanta', 'Miami']
    Detroit -> ['Chicago', 'Boston', 'Washington', 'New York']
    Philadelphia -> ['New York', 'Washington']
    Washington -> ['Atlanta', 'Miami', 'Detroit', 'Philadelphia']
    
    

## 4.3 최단경로
이 예제에선 이동별로 가중치가 없기 때문에 한 역에서 다른 역으로의 이동 시간을 최적화하기 위해 이동하는데 거치는 홉(hop)의 수가 중요하다.  

두 정점을 연결하는 에지의 집합을 경로(path)라 한다.
1. 경로는 에지 리스트를 취해서 연결된 정점을 파악한다.
2. 정점 목록을 유지하고 에지 리스트를 하나씩 제거하며 경로를 찾는다.


### 4.3.1 너비 우선 탐색
시작 지점과 목표 지점 사이의 에지가 가장 적은 경로를 찾는 것이다.  

Boston 에서 Miami 사이의 가장 짧은 길을 찾을 것이다.

이전 글 [Search Problems](https://moongni.github.io/algorithm/Search-Problems/#224-%EB%84%88%EB%B9%84-%EC%9A%B0%EC%84%A0-%ED%83%90%EC%83%89) 에서 구현한 `bfs()`, `Node`, `node_to_path()` 를 재사용할 수 있다.

`generic_search.py` 에서 구현한 `bfs()` 함수는 3개의 매개변수를 취한다.  

- initial: 시작지점 = "Boston"
- goal_test: Callable 함수로 목표지점이 맞는지 확인 = "Miami"인지 확인하는 lambda 함수
- successors: Callable 다음 이동지점 확인 함수 = neighbors_for_vertex() 메서드


```python
# graph.py 계속
# city_graph 변수에 2장의 너비 우선 탐색을 재사용
from generic_search import bfs, Node, node_to_path

bfs_result: Optional[Node[V]] = bfs("Boston", lambda x: x == "Miami",
                                    city_graph.neighbors_for_vertex)
if bfs_result is None:
    print("너비 우선 탐색으로 답을 찾을 수 없습니다.")
else:
    path: List[V] = node_to_path(bfs_result)
    print("Boston에서 Miami까지 최단 경로")
    print(path)
```

    Boston에서 Miami까지 최단 경로
    ['Boston', 'Detroit', 'Washington', 'Miami']
    

## 4.4 네트워크 구축 비용 최소화
예제의 미국의 15개 대도시 통계 구역을 연결하며 목표가 네트워크 구축 비용을 최소화라면, 최소한의 노선을 설치해야 할 것이다.  

### 4.4.1 가중치
노선의 양을 이해하려면 에지가 나타내는 거리를 알아야 한다. 이 예제의 가중치(weight)은 연결된 두 대도시 사이의 거리이다.  

가중치를 처리하기 위해 `Edge` 의 서브클래스 `WeightedEdge` 와 `Graph` 의 서브클래스 `WeightedGraph` 를 구현한다.  

프림(Prim)알고리즘(야르니크 알고리즘)에서 한 에지를 다른 에지와 비교하여 가장 작은 가중치를 가진 에지를 찾는 함수가 필요하다.


```python
# WeightedEdge.py
from __future__ import annotations
from dataclasses import dataclass
from edge import Edge


@dataclass
class WeightedEdge(Edge):
    weight: float

    def reversed(self) -> WeightedEdge:
        return WeightedEdge(self.v, self.u, self.weight)

    # 최소 가중치를 가진 에지를 찾기위해 가중치 순으로 정렬할 수 있음
    def __lt__(self, other: WeightedEdge) -> bool:
        return self.weight < other.weight

    def __str__(self) -> str:
        return f"{self.u} {self.weight} -> {self.v}"
```

`__lt__()` 메서드를 통해 < 연산을 구현할 때, 가중치에만 집중한다.  

`WeightedGraph` 클래스는 `Graph` 클래스의 여러 메서드를 상속받는다.
아래의 메서드를 추가한다.
- \__init__()
- WeightedEdge 객체를 추가하는 헬퍼 메서드
- neighbors_for_index_with_weights() 한 정점의 각 이웃과 에지의 가중치를 반환하는 메서드
- \__str()__


```python
# weighted_graph.py
from typing import TypeVar, Generic, List, Tuple
from graph import Graph
from weighted_edge impoprt WeightedEdge

V = TypeVar('V') # 그래프 정점(vertex) 타입

class WeightedGraph(Generic[V], Graph[V]):
    def __init__(self, vertices: List[V] = []) -> None:
        self._vertices: List[V] = vertices
        self._edges: List[List[WeightedEdge]] = [[] for _ in vertices]
        
    def add_edge_by_indices(self, u: int, v: int, weight: float) -> None:
        edge: WeightedEdge = WeightedEdge(u, v, weight)
        self.add_edge(edge) # Graph 클래스 메서드 호출
        
    def add_edge_by_vertices(self, first: V, second: V, weight: float) -> None:
        u: int = self._vertices.index(first)
        v: int = self._vertices.index(second)
        self.add_edge_by_indices(u, v, weight)
        
    def neighbors_for_index_with_weights(self, index: int) -> List[Tuple[V, float]]:
        distance_tuples: List[Tuple[V, float]] = []
        for edge in self.edges_for_index(index):
            distance_tuples.append((self.vertex_at(edge.v), edge.weight))
        return distance_tuples
    
    def __str__(self) -> str:
        desc: str = ""
        for i in range(self.vertex_count):
            desc += f"{self.vertex_at(i)} -> {self.neighbors_for_index_with_weights(i)}\n"
        return desc
```


```python
city_graph2: WeightedGraph[str] = WeightedGraph(["Seattle", "San Francisco", "Los Angeles", "Riverside",
                                                 "Phoenix", "Chicago", "Boston", "New York", "Atlanta",
                                                 "Miami", "Dallas", "Houston", "Detroit", "Philadelphia",
                                                 "Washington"])

city_graph2.add_edge_by_vertices("Seattle", "Chicago", 1737)
city_graph2.add_edge_by_vertices("Seattle", "San Francisco", 678)
city_graph2.add_edge_by_vertices("San Francisco", "Riverside", 386)
city_graph2.add_edge_by_vertices("San Francisco", "Los Angeles", 348)
city_graph2.add_edge_by_vertices("Los Angeles", "Riverside", 50)
city_graph2.add_edge_by_vertices("Los Angeles", "Phoenix", 357)
city_graph2.add_edge_by_vertices("Riverside", "Phoenix", 307)
city_graph2.add_edge_by_vertices("Riverside", "Chicago", 1704)
city_graph2.add_edge_by_vertices("Phoenix", "Dallas", 887)
city_graph2.add_edge_by_vertices("Phoenix", "Houston", 1015)
city_graph2.add_edge_by_vertices("Dallas", "Chicago", 805)
city_graph2.add_edge_by_vertices("Dallas", "Atlanta", 721)
city_graph2.add_edge_by_vertices("Dallas", "Houston", 225)
city_graph2.add_edge_by_vertices("Houston", "Atlanta", 702)
city_graph2.add_edge_by_vertices("Houston", "Miami", 968)
city_graph2.add_edge_by_vertices("Atlanta", "Chicago", 588)
city_graph2.add_edge_by_vertices("Atlanta", "Washington", 543)
city_graph2.add_edge_by_vertices("Atlanta", "Miami", 604)
city_graph2.add_edge_by_vertices("Miami", "Washington", 923)
city_graph2.add_edge_by_vertices("Chicago", "Detroit", 238)
city_graph2.add_edge_by_vertices("Detroit", "Boston", 613)
city_graph2.add_edge_by_vertices("Detroit", "Washington", 396)
city_graph2.add_edge_by_vertices("Detroit", "New York", 482)
city_graph2.add_edge_by_vertices("Boston", "New York", 190)
city_graph2.add_edge_by_vertices("New York", "Philadelphia", 81)
city_graph2.add_edge_by_vertices("Philadelphia", "Washington", 123)

print(city_graph2)
```

    Seattle -> [('Chicago', 1737), ('San Francisco', 678)]
    San Francisco -> [('Seattle', 678), ('Riverside', 386), ('Los Angeles', 348)]
    Los Angeles -> [('San Francisco', 348), ('Riverside', 50), ('Phoenix', 357)]
    Riverside -> [('San Francisco', 386), ('Los Angeles', 50), ('Phoenix', 307), ('Chicago', 1704)]
    Phoenix -> [('Los Angeles', 357), ('Riverside', 307), ('Dallas', 887), ('Houston', 1015)]
    Chicago -> [('Seattle', 1737), ('Riverside', 1704), ('Dallas', 805), ('Atlanta', 588), ('Detroit', 238)]
    Boston -> [('Detroit', 613), ('New York', 190)]
    New York -> [('Detroit', 482), ('Boston', 190), ('Philadelphia', 81)]
    Atlanta -> [('Dallas', 721), ('Houston', 702), ('Chicago', 588), ('Washington', 543), ('Miami', 604)]
    Miami -> [('Houston', 968), ('Atlanta', 604), ('Washington', 923)]
    Dallas -> [('Phoenix', 887), ('Chicago', 805), ('Atlanta', 721), ('Houston', 225)]
    Houston -> [('Phoenix', 1015), ('Dallas', 225), ('Atlanta', 702), ('Miami', 968)]
    Detroit -> [('Chicago', 238), ('Boston', 613), ('Washington', 396), ('New York', 482)]
    Philadelphia -> [('New York', 81), ('Washington', 123)]
    Washington -> [('Atlanta', 543), ('Miami', 923), ('Detroit', 396), ('Philadelphia', 123)]
    
    

### 4.4.2 최소 신장 트리 찾기
>**트리**란 두 정점 사이에 한 방향의 경로만 존재하는 그래프의 일종으로 **사이클(cycle)**이 없다는 것을 의미한다.  
그래프의 한 시작점에서 같은 에지를 반복하지 않고 다시 같은 시작점으로 돌아올 수 있다면 사이클이 존재하는 것이다.

최소 신장트리는 모든 정점을 연결한 신장 트리와 달리 모든 정점을 다룰 수 있는 최소 비용으로 연결한 트리이다.
최소한의 노선으로 설치하는 방법을 찾는 것은 최소 신장 트리를 찾는 것이다.

**우선순위 큐**
우선순위 큐 또한 이전 글 [Search Problems](https://moongni.github.io/algorithm/Search-Problems/#224-%EB%84%88%EB%B9%84-%EC%9A%B0%EC%84%A0-%ED%83%90%EC%83%89) 에서 구현한 `PriorityQueue` 클래스를 재사용한다.

#### 총 가중치 계산
최소 신장 트리를 구현하기 전 총 가중치를 계산하는 함수 `total_weight()` 을 구현한다. 


```python
#minimum_spanning_tree.py
from typing import TypeVar, List, Optional
from weighted_graph import WeightedGraph
from weighted_edge import WeightedEdge
from generic_search import PriorityQueue

V = TypeVar('V') # 그래프 정점(vertex) 타입
WeightedPath = List[WeightedEdge] # 경로 타입 앨리어스

def total_weight(wp: WeightedPath) -> float:
    return sum([e.weight for e in wp])
```

#### 프림(Prim) 알고리즘
최소 신장 트리를 찾기 위한 프림 알고리즘은 최소 신장 트리에 포함된 정점과 아직 포함하지 않은 정점으로 나누어 다음 4단계를 실행한다.
1. 최소 신장 트리에 포함할 한 정점을 정한다.
2. 아직 최소 신장 트리에 포함되지 않은 정점 중에서 정점에 연결된 가장 낮은 가중치 에지를 찾는다.
3. 가장 낮은 가중치 에지의 정점을 최소 신장 트리에 추가한다.
4. 그래프의 모든 정점이 최소 신장 트리에 추가될 때까지 2와 3을 반복한다.

프림 알고리즘은 우선순위 큐를 사용하여 새 정점이 최소 신장 트리에 추가될 때마다 트리 외부 정점에 연결되는 모든 출력 에지가 우선순위 큐에 추가된다.  

최소 가중치 에지는 우선순위 큐에서 `pop()` 되며, 알고리즘은 우선순위 큐가 빌 때까지 실행된다.

CAUTION: 프림 알고리즘은 방향이 있는 그래프에서 제대로 작동하지 않는다. 또한 연결되지 않은 그래프에서는 아예 작동하지 않는다.{: .notice--warning}


```python
#minimum_spanning_tree.py 계속

def mst(wg: WeightedGraph[V], start: int = 0) -> Optional[WeightedPath]:
    if start > (wg.vertex_count - 1) or start < 0: # 시작 정점이 유효한지 판단
        return None
    result: WeightPath = [] # 최소 신장 트리
    pq: PriorityQueue[WeightedEdge] = PriorityQueue()
    visited: [bool] = [False] * wg.vertex_count # 방문할 정점
    
    def visit(index: int):
        visited[index] = True # 방문한 정점으로 표시
        for edge in wg.edges_for_index(index):
            # 해당 정점의 모든 에지를 우선순위 큐(pq)에 추가
            if not visited[edge.v]:
                pq.push(edge)
    
    visit(start) # 첫 번째 정점에서 시작
    
    while not pq.empty: # pq에 에지가 없을 때까지 반복
        edge = pq.pop()
        if visited[edge.v]:
            continue # 방문한 곳이면 넘어감
        result.append(edge) # 최소 가중치 에지를 결과에 추가
        visit(edge.v) # 연결된 에지에 방문
        
    return result

def print_weighted_path(wg: WeightedGraph, wp: WeightedPath) -> None:
    for edge in wp:
        print(f"{wg.vertex_at(edge.u)} {edge.weight} > {wg.vertex_at(edge.v)}")
    print(f"가중치 총합 : {total_weight(wp)}")
```

>`mst()` 함수는 최소 신장 트리를 표현하는 `WeightedPath` 타입 앨리어스에서 선택된 경로를 반환한다.  

>`result` 변수에 최소 신장 트리의 가중치 경로를 저장한다. 최소 가중치의 에지를 `pop()`하여 이를 추가하고, 그래프의 다른 정점으로 이동한다.  
프림 알고리즘은 항상 최소 가중치의 에지를 선택하기 때문에 탐욕 알고리즘(greedy algorithm)이다.

>`visit()` 함수는 내부 헬퍼 함수로 방문한 정점을 표시하고, pq에 방문하지 않은 정점에 연결된 에지를 추가한다.


```python
result: Optional[WeightedPath] = mst(city_graph2)
if result is None:
    print("No solution found!")
else:
    print_weighted_path(city_graph2, result)
```

    Seattle 678 > San Francisco
    San Francisco 348 > Los Angeles
    Los Angeles 50 > Riverside
    Riverside 307 > Phoenix
    Phoenix 887 > Dallas
    Dallas 225 > Houston
    Houston 702 > Atlanta
    Atlanta 543 > Washington
    Washington 123 > Philadelphia
    Philadelphia 81 > New York
    New York 190 > Boston
    Washington 396 > Detroit
    Detroit 238 > Chicago
    Atlanta 604 > Miami
    가중치 총합 : 5372
    

위 결과는 가중치 그래프의 15개 도시를 최소 비용으로 연결하는 에지의 집합이다. 모든 노선을 연결하는 데 필요한 최소길이는 5,372마일이다.

## 4.5 가중치 그래프에서 최단 경로 찾기
가중치가 있는 네트워크의 한 정점에서 다른 정점까지의 최단경로를 찾는 알고리즘

### 4.5.1 다익스트라 알고리즘
다익스트라 알고리즘은 가중치 그래프에서 최단 경로를 반환한다. 단일 소스 정점에서 가장 가까운 정점을 계속 탐색한다.  
>프림 알고리즘과 동일하게 탐욕적이다.

다익스트라 알고리즘의 과정
1. 시작 정점을 우선순위 큐에 추가한다.
2. 우선순위 큐에서 가장 가까운 정점을 팝한다.
3. 현재 정점에서 연결된 모든 이웃 정점을 확인하여 정점의 거리와 에지를 기록하고 우선순위 큐에 추가한다.
4. 우선순위 큐가 빌 때까지 2,3을 반복한다.
5. 시작점에서 다른 모든 정점까지의 최단 거리를 반환한다.

`다익스트라 노드(DijkstraNode)` 클래스를 구현하여 각 정점과 해당 비용을 추적하고 비교한다. 2장의 [Node](https://moongni.github.io/algorithm/Search-Problems/#223-%EA%B9%8A%EC%9D%B4-%EC%9A%B0%EC%84%A0-%ED%83%90%EC%83%89) 클래스와 비슷하다.


```python
#dijkstra.py
from __future__ import annotations
from typing import TypeVar, List, Optional, Tuple, Dict
from dataclasses import dataclass
from mst import WeightedPath, print_weighted_path
from weighted_graph import WeightedGraph
from weighted_edge import WeightedEdge
from priority_queue import PriorityQueue

V = TypeVar('V') # 그래프 정점(vertex) 타입

@dataclass
class DijkstraNode:
    vertex: int
    distance: float
    
    def __lt__(self, other: DijkstraNode) -> bool:
        return self.distance < other.distance
    
    def __eq__(self, other: DijkstraNode) -> bool:
        return self.distance == other.distance

def dijkstra(wg: WeightedGraph[V], root: V) -> Tuple[List[Optional[float]], Dict[int, WeightedEdge]]:
    first: int = wg.index_of(root) # 시작 인덱스 찾기
    distances: List[Optional[float]] = [None] * wg.vertex_count
    distances[first] = 0 # 자기자신을 향한 거리는 0
    path_dict: Dict[int, WeightedEdge] = {} # 정점에 대한 경로
    pq: PriorityQueue[DijkstraNode] = PriorityQueue()
    pq.push(DijkstraNode(first, 0))
    
    while not pq.empty:
        u: int = pq.pop().vertex # 다음 가까운 정점을 탐색
        dist_u: float = distances[u] # 이 정점에 대한 거리를 이미 알고있음
        
        # 이 정점에서 모든 에지 및 정점을 살펴본다.
        for we in wg.edges_for_index(u):
            # 이 정점에 대한 이전 거리
            dist_v: float = distances[we.v]
            # 이전 거리가 없거나 혹은 최단 경로가 존재한다면
            if dist_v is None or dist_v > we.weight + dist_u:
                # 정점 거리 갱신
                distances[we.v] = we.weight + dist_u
                # 정점의 최단 경로 에지를 갱신
                path_dict[we.v] = we
                # 해당 정점을 우선순위 큐에 넣는다.
                pq.push(DijkstraNode(we.v, we.weight + dist_u))
    return distances, path_dict

# 다익스트라 알고리즘 결과를 더 쉽게 접근하게 하는 헬퍼 함수
def distance_array_to_vertex_dict(wg: WeightedGraph[V], distances: List[Optional[float]]) -> Dict[V, Optional[float]]:
    distance_dict: Dict[V, Optional[float]] = {}
    for i in range(len(distances)):
        distance_dict[wg.vertex_at(i)] = distances[i]
    return distance_dict

# 에지의 딕셔너리 인자를 취해 각 노드에 접근,
# 정점 start에서 end까지 가는 에지 리스트를 반환한다.
def path_dict_to_path(start: int, end: int, path_dict: Dict[int, WeightedEdge]) -> WeightedPath:
    if len(path_dict) == 0:
        return []
    
    edge_path: WeightedPath = []
    e: WeightedEdge = path_dict[end]
    edge_path.append(e)
    while e.u != start:
        e = path_dict[e.u]
        edge_path.append(e)
    return list(reversed(edge_path))
```

>`while not pq.empty`  
우선순위 큐가 빌 때까지 다익스트라 알고리즘을 계속 실행한다. 변수 `u`는 현재 정점이고, `dist_u` 는 `u` 에 대한 거리이다.  

>`for we in wg.edges_for_index(u):`  
변수 `u` 에 연결된 모든 에지를 탐색하고 이 단계에서 탐색한 모든 정점은 거리가 갱신된다.


```python
distances, path_dict = dijkstra(city_graph2, "Los Angeles")
name_distance: Dict[str, Optional[int]] = distance_array_to_vertex_dict(city_graph2, distances)
print("Los Angeles에서부터 각 정점의 거리:")
for key, value in name_distance.items():
    print(f"{key} : {value}")
print()

print("Los Angeles에서 Boston까지 최단 경로:")
path: WeightedPath = path_dict_to_path(city_graph2.index_of("Los Angeles"), city_graph2.index_of("Boston"), path_dict)
print_weighted_path(city_graph2, path)
```

    Los Angeles에서부터 각 정점의 거리:
    Seattle : 1026
    San Francisco : 348
    Los Angeles : 0
    Riverside : 50
    Phoenix : 357
    Chicago : 1754
    Boston : 2605
    New York : 2474
    Atlanta : 1965
    Miami : 2340
    Dallas : 1244
    Houston : 1372
    Detroit : 1992
    Philadelphia : 2511
    Washington : 2388
    
    Los Angeles에서 Boston까지 최단 경로:
    Los Angeles 50 > Riverside
    Riverside 1704 > Chicago
    Chicago 238 > Detroit
    Detroit 613 > Boston
    가중치 총합 : 2605
    

다익스트라 알고리즘과 프림 알고리즘은 둘 다 탐욕 알고리즘으로 비슷한 코드를 사용하여 구현할 수 있다.  
또한 2장의 $A^*$ 알고리즘과 닮았다. 다익스트라 알고리즘에서 단일 대상을 찾도록 제한하고, 휴리스틱을 추가하면 두 알고리즘은 동일하다.  

NOTE: 위의 다익스트라 알고리즘은 양수 가중치 그래프를 다루도록 설계되었다. 음수 가중치 그래프를 다루려면 수정하거나 대체 알고리즘을 사용해야 한다.{: .notice--public}