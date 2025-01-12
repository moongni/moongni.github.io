---
layout: single
title: "[Node.js] 파일 시스템"
categories: Node.js
tags: [JavaScript]
classes: wide
author_profile: false
sidebar:
    nav: "docs"
use_math: true
---
# 파일
노드의 파일시스템은 파일을 관리와 디렉터리 관리로 구성되어있고 동기식, 비동기식 IO 기능을 제공한다.  
동기식 IO 메소드는 `Sync` 라는 단어를 붙인다.


```javascript
var fs = require('fs');

// 파일을 동기식 io로 읽어 들인다.
var data = fs.readFileSync('./output.txt', 'utf8');

// 읽어 들인 데이터를 출력
console.log(data);
```

    Hello World!
    


```javascript
var fs = require('fs');

// 파일을 비동기식 io로 읽기
fs.readFile('./output.txt', 'utf8', function(err, data){
    //읽은 데이터 출력
    console.log(data);
});

console.log('프로젝트 폴더 안의 package.json 파일을 읽도록 요청');
```

    프로젝트 폴더 안의 package.json 파일을 읽도록 요청
    Hello World!
    

비동기식io 방식이 노드에서 더욱 자주 사용되며 `readFile()` 메소드를 실행시에 파일을 읽는 작업이 끝난 후에 `function` 함수가 호출되며 `fuction` 함수의 `err` 는 오류가 발생하면 `err` 에 오류데이터가 들어가고 아니면 `null` 값을 갖는다. 일반적으로 첫번째 파라미터인 err가 null 값인지 체크하는 코드를 사용한 후에 문제가 없으면 파읽 읽기에 성공한 것으로 처리한다.

파일을 읽는 것뿐만 아니라 쓰는 기능까지 `fs` 모듈에 존재한다.

메소드|설명
:---|---
readFile(filename, [encoding], [callback])|비동기식io로 파일 읽기
readFileSync(filename, [encoding])|동기식 io로 파일 읽기
writeFile(filename, data, encoding='utf8', [callback])|비동기식 io로 파일 쓰기
writeFileSync(filename, data, encoding='utf8')|동기식 io로 파일 쓰기


```javascript
//비동기 파일 쓰기
var fs = require('fs');

fs.writeFile('./output.txt', 'Hello World!!', function(err){
    if(err){
        console.log('Err : '+err);
    }
    console.log('output.txt 데이터 쓰기 완료');
});
console.log('프로젝트 폴더 안의 output.txt 파일을 쓰기 요청');

fs.readFile('./output.txt', 'utf8', function(err, data){
    if(err){
        console.log('err : ' + err);
    }
    console.log(data);
})
```

    프로젝트 폴더 안의 output.txt 파일을 쓰기 요청
    output.txt 데이터 쓰기 완료
    Hello World!!
    


```javascript
//동기식 파일 쓰기
var fs = require('fs');
fs.writeFileSync('./outputsync.txt','Hello World!');
console.log('프로젝트 폴더 안의 output.txt 파일을 쓰기 요청');
var data = fs.readFileSync('./outputsync.txt', 'utf8');
console.log(data);
console.log('프로젝트 폴더 안의 output.txt 파일을 읽기 요청');
```

    프로젝트 폴더 안의 output.txt 파일을 쓰기 요청
    Hello World!
    프로젝트 폴더 안의 output.txt 파일을 읽기 요청
    

## 직접 열고 닫으면서 읽기
파일을 읽을 때 조금씩 읽거나 조금씩 쓰는 경우 , 다른 곳에서 받아 온 데이터를 파일에 쓰는 경우도 있기 때문에 파일을 다루는 다양한 방식이 정의되어있다.

메소드|설명|
:---|---
open(path, flags, [mode], [callback])| 파일 열기
read(fd, buffer, offset, length, position, [callback])| 지정부분 파일 읽기
write(fd, buffer, offset, length, position, [callback])| 지정부분 데이터 쓰기
close(fd, [callback])| 파일 닫기


```javascript
var fs = require('fs');

//데이터 쓰기
fs.open('./output.txt', 'w', function(err, fd){
    if(err) throw err;
    
    var buf = new Buffer('hi!!\n');
    fs.write(fd, buf, 0, buf.length, null, function(err, written, buffer){
        if(err) throw err;
        
        console.log(err, written, buffer);
        
        fs.close(fd, function(){
            console.log('파일열고 데이터 쓰고 파일 닫기');
        });
    });
});
```

    null 5 <Buffer 68 69 21 21 0a>
    파일열고 데이터 쓰고 파일 닫기
    

함수를 호출할 때마다 파라미터로 함수를 전달하기 때문에 `open`->`write`->`close` 함수를 호출하는 순서가 중요하다.

`open()` 메소드를 호출할 때 플래그 값으로는 대표적으로 r, w, w+, a+ 가 있다. 

플래그|설명
:---|---
'r'|읽기, 파일이 없을 시에 예외발생
'w'|쓰기, 파일이 없을 시 만들고, 있으면 이전 내용을 삭제
'w+'|일기와 쓰기 모두 사용, 파일이 없을 시 만들고 파일이 있으면 이전 내용을 삭제
'a+'|읽기와 추가외 모두 사용, 파일 없으면 시 만들고 파일이 있으면 이전 내용에 새로운 내용 추가


```javascript
var fs = require('fs');

//데이터 읽기 'r'
fs.open('./output.txt', 'r', function(err, fd){
    if(err) throw err;
    
    var buf = new Buffer(10);
    console.log('buffer type: %s', Buffer.isBuffer(buf));
    
    fs.read(fd, buf, 0, buf.length, null, function(err, bytesRead, buffer){
        if(err) throw err;
        
        var inStr = buffer.toString('utf8', 0, bytesRead);
        console.log('읽은 데이터 : %s', inStr);
        
        console.log(err, bytesRead, buffer);
        
        fs.close(fd, function(){
            console.log('output.txt 파일을 열고 읽기');
        });
    });
});
```

    buffer type: true
    읽은 데이터 : hi!!
    
    null 5 <Buffer 68 69 21 21 0a 00 00 00 00 00>
    output.txt 파일을 열고 읽기
    

## Buffer 객체 사용


```javascript
// 버퍼 객체를 크기만 지정하여 만든 후 문자열 쓰기
var output = '안녕1 !';
var buffer1 = new Buffer(10);
var len = buffer1.write(output, 'utf8');
console.log('첫 번째 버퍼의 문자열: %s', buffer1.toString());

// 버퍼 객체를 문자열을 이용해 만들기
var buffer2 = new Buffer('안녕2 !', 'utf8');
console.log('두 번째 문자열 : %s', buffer2.toString());

// 타입확인
console.log('버퍼 객체의 타입 : %s', Buffer.isBuffer(buffer1)); //true

// 버퍼 객체에 들어 있는 문자열 데이터를 문자열 변수로 만들기
var byteLen = Buffer.byteLength(output);
var str1 = buffer1.toString('utf8', 0, byteLen);
var str2 = buffer2.toString('utf8');

// 첫 번째 버퍼에 있는 데이터 두 번째 버퍼 객체로 복사
buffer1.copy(buffer2, 0, 0, len);
console.log('두 번째 버퍼에 복사한 후의 문자열 : %s', buffer2.toString('utf8'));

// 두 개의 버퍼를 붙임
var buffer3 = Buffer.concat([buffer1, buffer2]);
console.log('두 버퍼를 붙인 문자열 : %s', buffer3.toString('utf8'));
```

    첫 번째 버퍼의 문자열: 안녕1 ! 
    두 번째 문자열 : 안녕2 !
    버퍼 객체의 타입 : true
    두 번째 버퍼에 복사한 후의 문자열 : 안녕1 !
    두 버퍼를 붙인 문자열 : 안녕1 ! 안녕1 !
    

## 스트림 단위로 파일 읽고 쓰기
파일을 읽거나 쓸 때는 데이터 단위가 아닌 스트림 단위로 처리할 수도 있다. 파일에서 읽을 때는 `createReadStream()` , 파일을 쓸 때는 `createWriteStream()` 메소드로 스트림 객체를 만든 후 데이터를 읽고 쓴다.

메소드|설명
:---|---
createReadStream(path, [options])|파일을 읽기 위한 스트림객체 생성
createWriteStream(path, [options])|파일을 쓰기 위한 스트림객체 생성

스트림은 Node.js 애플리케이션을 구동하는 기본 개념 중 하나로, 파일 읽기/쓰기, 네트워크 통신 등을 포함한 모든 종단 간 정보 교환을 효율적인 방식으로 처리하는 방법니다.  

예를들어, 과거 프로그램에 파일을 읽도록 명령하면 파일을 처음부터 끝까지 메모리로 다 읽은 후 처리했다.
반면 스트림을 사용하면 한 조각씩 읽고 모든 내용을 메모리에 유지하지 않고 내용을 처리할 수 있다.

이는 두가지 이점을 준다.  
- 메모리 효율성: 처리하기 전에 메모리에 많은 양의 데이터를 로드할 필요가 없다.
- 시간 효율성: 전체 데이터 페이로드를 기다리지 않고 데이터가 있는 즉시 처리를 시작할 수 있으므로 데이터 처리를 시작하는데 시간이 덜 걸린다.


```javascript
var fs = require('fs');

var infile = fs.createReadStream('./output.txt', {flags:'r'});
var outfile = fs.createWriteStream('./output2.txt', {flags:'w'});

infile.on('data', function(data){
    console.log('읽어들인 데이터 :', data);
    console.log(data.toString())
    outfile.write(data);
});

infile.on('end', function(){
    console.log('파일 읽기 종료');
    outfile.end(function(){
        console.log('파일 쓰기 종료');
    });
});
```



    읽어들인 데이터 : <Buffer 68 69 21 21 0a>
    hi!!
    
    파일 읽기 종료
    파일 쓰기 종료
    


```javascript
var infile = fs.createReadStream('./output2.txt', {flags: 'r'});

infile.on('data', function(data){
    console.log(data.toString());
})
```



    hi, lets go burry
    
    

두개의 스트림을 붙여주면 더 간단히 만들 수 있다. `pipe()` 메소드는 두개의 스트림을 붙여준다.


```javascript
var fs = require('fs')

var inname = './output.txt';
var outname = './output2.txt';

fs.exists(outname, function(exists){
    if(exists){
        fs.unlink(outname, function(err){
            if(err) throw err;
            console.log('기존파일 [' + outname +'] 삭제');
        });
    }
    var infile = fs.createReadStream(inname, {flags: 'r'});
    var outfile = fs.createWriteStream(outname, {flags: 'w'});
    infile.pipe(outfile);
    console.log('파일복사 ['+inname+']-> ['+outname+']');
});
```

    파일복사 [./output.txt]-> [./output2.txt]
    기존파일 [./output2.txt] 삭제
    

## http 모듈로 요청받은 파일내용을 읽고 응답
스트림 연결방법은 웹 서버를 만들고 사용자의 요청을 처리할 때 유용하다. 다음은 http모듈을 사용해서 사용자로부터 요청을 받았을 때 파일의 내용을 읽어 응답으로 보내는 코드이다.


```javascript
var fs = require('fs');
var http = require('http');
var server = http.createServer(function(req, res){
    // 파일을 읽어 응답 스트림과 pipe로 연결
    var instream = fs.createReadStream('./output.txt');
    instream.pipe(res);
})
server.listen(7001, '127.0.0.1');
```



웹 서버에서 클라이언트로부터 요청을 받으면 먼저 output.txt 파일에서 스트림을 만든 후 클라이언트로 데이터를 보낼 수 있는 스트림과 연결해줄 수 있다. 두 객체의 연결은 파일을 읽는 것도 `Stream` 객체 이고, 데이터를 쓰기 위해 웹 서버에서 클라이언트 쪽에 만든 것도 `Stream` 객체 이기 때문이다.

## fs 모듈로 새 디렉터리 만들고 삭제하기



```javascript
var fs = require('fs');
fs.mkdir('./docs', 0666, function(err){
    if(err) throw err;
    console.log('새로운 docs 폴더 생성');

    fs.rmdir('./docs', function(err){
        if(err) throw err;
        console.log('docs 폴더 삭제');
    });
});
```

    새로운 docs 폴더 생성
    docs 폴더 삭제
    

파일의 내용을 읽거나 쓰는 간단한 자바스크립트 코드여도 노드의 비동기 프로그래밍 방식이 콜백 함수를 사용하기 때문에 코드 구조가 조금 복잡해 보일 수 있다.
