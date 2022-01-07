---
layout: single
title: "[Node.js] Request Parameter"
categories: Node.js
tags: [Node.js, JavaScript]
classes: wide
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---
# [Node.js] 주소문자열과 요청 파라미터

## 주소 문자열과 요청 파라미터
웹 사이트에 접속하기 위한 사이트 주소 정보는 노드에서 `URL` 객체로 만들 수 있다.
google에서 workout를 검색하면 다음과 같은 주소 문자열을 만들어 검색 요청을 한다.  
`https://www.google.com/search?q=workout&oq=workout&aqs=chrome..69i57j69i59j69i60l2.1246j0j9&sourceid=chrome&ie=UTF-8`
  
이렇게 만들어진 주소 문자열은 단순 문자열이므로 서버가 이 정보를 받아 처리할 때, 어디까지가 사이트 주소인지, 어떤 내용이 파라미터인지 구별해야 한다. 이 구별을 위해서 `?` 기호를 기준으로 앞과 뒤의 문자열을 분리하는 경우가 많다. 이 작업을 쉽게 하기 위해 노드에서 만들어 둔 모듈이 `url` 모듈이다.  
  
### 주소 문자열을 URL 객체로 변환

|메소드|설명|
|:---|---|
|parse() | 주소 문자열을 파싱하여 URL객체로 변환|
|format() | URL 객체를 주소 문자열로 변환|



```javascript
//주소 문자열을 URL 객체로 변환
var url = require('url');

//parse()
var curURL = url.parse('https://www.google.com/search?q=workout&oq=workout&aqs=chrome..69i57j69i59j69i60l2.1246j0j9&sourceid=chrome&ie=UTF-8');

//format()
var curStr = url.format(curURL);

console.dir(curURL);
console.log('주소 문자열 : %s', curStr);
```

    Url {
      protocol: 'https:',
      slashes: true,
      auth: null,
      host: 'www.google.com',
      port: null,
      hostname: 'www.google.com',
      hash: null,
      search: '?q=workout&oq=workout&aqs=chrome..69i57j69i59j69i60l2.1246j0j9&sourceid=chrome&ie=UTF-8',
      query: 'q=workout&oq=workout&aqs=chrome..69i57j69i59j69i60l2.1246j0j9&sourceid=chrome&ie=UTF-8',
      pathname: '/search',
      path: '/search?q=workout&oq=workout&aqs=chrome..69i57j69i59j69i60l2.1246j0j9&sourceid=chrome&ie=UTF-8',
      href: 'https://www.google.com/search?q=workout&oq=workout&aqs=chrome..69i57j69i59j69i60l2.1246j0j9&sourceid=chrome&ie=UTF-8'
    }
    주소 문자열 : https://www.google.com/search?q=workout&oq=workout&aqs=chrome..69i57j69i59j69i60l2.1246j0j9&sourceid=chrome&ie=UTF-8
    

### 요청 파라미터 확인
URL 객체 속성의 주소 문자열의 query 속성은 요청 파라미터 정보를 가지고 있다. 여러개의 파라미터가 모두 들어있는 query를 웹 서버에서 각각 요청 파라미터로 분리해한다.  
```
query: 'q=workout&oq=workout&aqs=chrome..69i57j69i59j69i60l2.1246j0j9&sourceid=chrome&ie=UTF-8'
```
요청 파라미터는 `&` 기호로 구분 되어진다. `querystring` 모듈을 통해 분리 할 수 있다.  

|메소드|설명|
|:---|---:|
|parse()|요청 파라미터 문자열을 파싱하여 요청 파라미터 객체로 변환|
|stringfy()|요청 파라미터 객체를 문자열로 변환|


```javascript
//querystring
var querystring = require('querystring');
var para = querystring.parse(curURL.query);

console.log('query: %s' , para.q);
console.log('원래 요청 파라미터 : %s', querystring.stringify(param));
```

    query: workout
    원본 요청 파라미터 : q=workout&oq=workout&aqs=chrome..69i57j69i59j69i60l2.1246j0j9&sourceid=chrome&ie=UTF-8
    
