# Restful API
## Restful架构
- REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构
- RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践
- RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好

## RESTful API 请求设计
### 五种HTTP方法 
    # 列出所有动物园 get方法
    /zoos 
    # 新建一个动物园 post方法
    /zoos
    # 获取某个指定动物园的信息 get方法
    /zoos/:id
    # 更新某个指定动物园的全部信息 put方法
    /zoos/:id
    # 更新某个指定动物园的部分信息 patch方法
    /zoos/:id
    # 删除某个动物园 delete方法
    /zoos/:id
    # 列出某个指定动物园的所有动物 get方法
    /zoos/:id/animals
    # 删除某个指定动物园得到指定动物 delete方法
    /zoos/:id/animals/:id
过滤信息，如果记录数量很多， API应该提供参数，过滤返回结果 **?limit=10** 指定返回记录的数量 **?offset=10** 指定返回记录的开始位置

### RESTful API 响应设计
#### HTTP 状态码
- 1XX 相关信息
- 2XX 操作成功
- 3XX 重定向
- 4XX 客户端错误
- 5XX 服务器错误
客户端的每一次请求，服务器都必须给出回应。回应包括 HTTP 状态码和数据两部分。
五大类状态码，总共100多种，覆盖了绝大部分可能遇到的情况。每一种状态码都有约定的解释，客户端只需查看状态码，就可以判断出发生了什么情况。API 不需要1xx状态码。

#### 服务器回应数据
- 客户端请求时，要明确告诉服务器，接受 JSON 格式，请求的 HTTP 头的 ACCEPT 属性要设成 application/json
- 服务端返回的数据，不应该是纯文本，而应该是一个 JSON 对象。服务器回应的 HTTP 头的 Content-Type 属性要设为 application/json
- 错误处理 如果状态码是4xx，就应该向用户返回出错信息。一般来说，返回的信息中将 error 作为键名，出错信息作为键值即可。 {error: "Invalid API key"}
- 认证 RESTful API 应该是无状态，每个请求应该带有一些认证凭证。推荐使用 JWT 认证，并且使用 SSL
- Hypermedia 即返回结果中提供链接，连向其他API方法，使得用户不查文档，也知道下一步应该做什么

## API请求
1. HTTP 动词
    GET：   读取（Read）  请求指定的资源，被认为是不安全的方法
    POST：  新建（Create） 会向服务器提交数据，请求服务器进行处理 例如提交表单、文件上传
    PUT：   更新（Update）  会向服务器提交数据，进行数据修改
    PATCH： 更新（Update），通常是部分更新  
    DELETE：删除（Delete） 请求服务器删除所请求的URI
    HEAD : 与GET方法一样，但服务器不会回传响应主体，一般用来测试服务器性能
GET请求会将表单的数据以键值对的形式拼接到URL后发送给服务器。这就是GET和POST最重要的区别，对于一些敏感信息，GET方法是及其不安全的，千万不要这样使用。

2. URL（宾语）必须是名词
宾语就是 API 的 URL，是 HTTP 动词作用的对象。它应该是名词，不能是动词。比如，/articles这个 URL 就是正确的，而下面的 URL 不是名词，所以都是错误的。

    /getAllCars
    /createNewCar
    /deleteAllRedCars
既然 URL 是名词，为了统一起见，建议都使用复数。



3. 举个例子
    GET    /zoos：列出所有动物园
    POST   /zoos：新建一个动物园
    GET    /zoos/ID：获取某个指定动物园的信息
    PUT    /zoos/ID：更新某个指定动物园的信息（提供该动物园的全部信息）
    PATCH  /zoos/ID：更新某个指定动物园的信息（提供该动物园的部分信息）
    DELETE /zoos/ID：删除某个动物园
    GET    /zoos/ID/animals：列出某个指定动物园的所有动物
    DELETE /zoos/ID/animals/ID：删除某个指定动物园的指定动物


4. 过滤信息（Filtering）
如果记录数量很多，服务器不可能都将它们返回给用户。API应该提供参数，过滤返回结果。
    下面是一些常见的参数。

    ?limit=10：指定返回记录的数量
    ?offset=10：指定返回记录的开始位置。
    ?page=2&per_page=100：指定第几页，以及每页的记录数。
    ?sortby=name&order=asc：指定返回结果按照哪个属性排序，以及排序顺序。
    ?animal_type_id=1：指定筛选条件
    #参数的设计允许存在冗余，即允许API路径和URL参数偶尔有重复。比如，GET /zoo/ID/animals 与 GET /animals?zoo_id=ID 的含义是相同的。



5. 不符合 CRUD 情况的 RESTful API
在实际资源操作中，总会有一些不符合 CRUD（Create-Read-Update-Delete） 的情况，一般有几种处理方法。

- 使用 POST，为需要的动作增加一个 endpoint，使用 POST 来执行动作，比如: POST /resend 重新发送邮件。

- 增加控制参数，添加动作相关的参数，通过修改参数来控制动作。比如一个博客网站，会有把写好的文章“发布”的功能，可以用上面的 POST /articles/{:id}/publish 方法，也可以在文章中增加 published:boolean 字段，发布的时候就是更新该字段 PUT /articles/{:id}?published=true

- 把动作转换成资源，把动作转换成可以执行 CRUD 操作的资源， github 就是用了这种方法。比如“喜欢”一个 gist，就增加一个 /gists/:id/star 子资源，然后对其进行操作：“喜欢”使用PUT /gists/:id/star，“取消喜欢”使用 DELETE /gists/:id/star。

- 另外一个例子是 Fork，这也是一个动作，但是在 gist 下面增加 forks资源，就能把动作变成 CRUD 兼容的：POST /gists/:id/forks 可以执行用户 fork 的动作。


6. 动词覆盖，应对服务器不完全支持 HTTP 的情况
有些客户端只能使用GET和POST这两种方法。服务器必须接受POST模拟其他三个方法（PUT、PATCH、DELETE）。

这时，客户端发出的 HTTP 请求，要加上X-HTTP-Method-Override属性，告诉服务器应该使用哪一个动词，覆盖POST方法。


## API响应


## JWT
JSON Web Token 通过数字签名的方式，以JSON对象为载体，在不同的服务终端之间安全的传输信息
JWT 最常见的场景就是授权认证，一旦用户登陆，后续每个请求都将包含JWT，系统在每次处理用户请求之前，都将先进行JWT安全校验，通过之后再进行处理。

