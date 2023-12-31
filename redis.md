## Redis
- 与传统数据库不同的是Redis的数据是存在内存中的(内存数据库)， 读写速度非常快，被广泛应用于缓存方向，存储的是KV键值对数据
## Redis为什么这么快?
- Redis基于内存，内存的访问速度是磁盘的上千倍
- 内置了多种优化过后的数据结构实现，性能非常高
- 基于Reactor模式开发了一套高效的事件处理模型，主要是**单线程事件循环和IO多路复用**
## Redis的优点
- 支持丰富的数据类型，支持更复杂的应用场景
- 支持数据的持久化，可以将内存数据保存在磁盘中，重启后可以再次使用
- 有容灾恢复机制
- 在服务器内存使用完之后，可以将不用的数据存储在磁盘中
- 支持Cluster模式
- 使用单线程的多路IO复用模型
- 支持发布订阅模型，Lua脚本和事务等功能
- 使用了惰性删除与定期删除
## 为什么要用Redis(缓存)?
- 高性能，保证用户访问数据能直接从缓存中获取，提高效率
- 高并发，直接操作缓存能够承受的数据库请求远远大于直接访问数据库的，将数据库中的部分数据转移到缓存中去，这样用户请求会直接到缓存不到数据库中去，提高系统的整体并发
## Redis除了做缓存还能做什么？
- 分布式锁
- 限流
- 消息队列
- 复杂业务场景
## Redis做消息队列，不是特别建议Redis做消息队列
- 数据结构Stream做消息队列，支持
  - 发布订阅模式
  - 按照消费者组进行消费
  - 消息持久化(RDB AOF)
## Redis数据结构
- 5种基础的数据结构: String  List(双端链表)  Set(哈希表)   Hash   Zset(压缩列表  跳表)
- 3种特殊数据结构: HyperLogLogs(基数统计)  Bitmap(位存储)  Geospatial(地理位置)  Stream(消息队列) 自动生成全局唯一消息ID 支持以消费者组形式消费数据
  
## Redis的数据结构底层实现方式
- String: 底层实现方式为简单动态字符串(SDS)
  - **SDS不仅可以作为字符串，还可以记录字符串的长度，以及可以保存二进制数据**
  - SDS获取字符串长度的时间复杂度为O(1)，**SDS结构里面用len字段记录了字符串的长度**
  - 字符串的截取、拼接、查找、计数等操作，Redis的底层都是通过SDS的API实现的
- List: **底层实现方式为双向链表或者压缩列表**
  - 列表元素个数小于512个同时列表每个元素值都小于64字节，使用压缩列表，反之用双向列表
  - redis3.2之后，统一使用quicklist
- Hash: 底层实现方式为**哈希表或者压缩列表**
  - 哈希元素个数小于512个同时每个元素值都小于64字节，使用压缩列表，反之用哈希表
  - redis7.0之后，统一使用listpack
- Set: 底层实现方式为**哈希表或者整数集合**
  - 集合中元素都是整数且元素个数小于512， 会使用整数集合，反之使用哈希表
- Zset: 底层实现方式为压缩列表或者跳表
- redis7.0中，压缩列表已经被废弃，交由listpack代替
## String的应用场景
- 常规数据的缓存
- 技术比如用户单位时间的请求数，页面单位时间的访问数
- 分布式锁
  - redis方便进行存储数据的删除 key-value方便进行删除、添加以及设置过期时间
  - 添加锁的时候，key为锁的名称，而value为客户端生成的UUID标识符，同时设置过期时间
  - 当持锁的时间超过了锁的过期时间，会自动释放锁，同时主动释放锁的时候，需要判断本线程是否持有该锁，防止锁的错误释放，当UUID标识符对应上已经是持有锁的线程进行释放，则成功释放分布式锁
- 共享Seesion
  - 由于分布式服务器集群的存在，多次不同的访问服务器，导致Session只存在一个服务器上，导致下次访问服务器，被系统分配到别的没有存储Session的服务器，必须再次登录。可以将Session存在每个服务器上，此方法比较麻烦，采用Redis存储Session，所有服务器共用同一个Redis服务器，保证Session的一致性
## String还是Hash存储对象数据更好呢?
- String存储的是序列化后的对象数据，存放的是整个对象，Hash是对对象的每个字段单独存储，可以获取部分字段的信息，可以修改或者添加部分字段，如果需要对字段进行处理，那么使用Hash更好
- String存储相对更加节省内存，缓存相同数量的对象数据，String消耗的内存约是Hash的一半，绝大部分情况下，使用String存储对象数据即可
## String的底层实现
- SDS实现String，Redis会根据初始化的长度决定使用哪种类型，减少内存的使用
  - len: 字符串的长度也就是使用的字节数
  - alloc: 总共可用的字符空间大小
  - buf[]: 实际存储字符串的数据
  - flags: 低三位保存标志
- SDS的提升
  - 可以避免缓冲区溢出
  - 获取字符串长度的复杂度较低
  - 减少内存分配次数
  - 二进制安全
## 购物车信息用String还是Hash存储更好呢？
- 由于购物车种的商品频繁修改和变动，购物车信息建议使用Hash存储，用户ID为key 商品ID为field 商品数量为value
## 使用Redis做一个排行榜怎么做？
- Redis种有一个叫做sorted set的数据结构经常被用在排行榜种，ZRANGE(从小排到大)    ZREVRANGE(从大排到小) ZREVRANK(指定元素排名)
## 使用Set实现抽奖系统需要用到什么命令?
- SPOP key count 随机移除并获取指定集合中的一个或多个元素，适合不允许重复中奖的场景
- SRANDMEMBER key count 随意获取指定集合中的一个或多个元素，适合允许重复中奖的场景
## 使用Bitmap统计活跃用户怎么做？
- 使用日期（精确到天）作为key,用户ID作为offset，如果当日活跃过的就设置为1
## 使用HyperLogLog统计页面UV怎么做？
- 统计注册IP数 每日访问IP数 页面实时UV 在线用户数 共同好友数
# Redis线程模型
- 对于读写命令，Redis一直是单线程模型，在Redis4.0之后引入多线程执行一些大键值对的异步删除操作，6.0版本之后引入多线程来处理网络请求(提高网络IO读写性能)
- Redis基于Reactor模式设计开发了高效的事件处理模型，对应的是Redis中的文件事件处理器，它是单线程方式运行的，所以一般说Redis是单线程模型
- 文件事件处理器是单线程方式运行的，但通过IO多路复用程序来监听多个套接字，实现了高性能的网络通信模型，又很好地与Redis服务器中其他同样以单线程方式运行的模型进行对接，保证了单线程设计的简单性
- Redis通过IO多路复用来监听来客户端的大量请求，将感兴趣的事件和类型注册到内核并监听每个事件的发生，IO多路复用技术的使用让Redis不需要额外创建多余的线程来监听客户端的大量链接，降低资源的消耗。
- 文件事件处理器
  - 多个Socket (客户端连接)
  - IO多路复用程序(支持多个客户端连接的关键)
  - 文件事件分派器(将socket关联到相应的事件处理器)
  - 事件处理器(连接应答处理器，命令请求处理器，命令恢复处理器)
- redis6.0版本之后，在redis启动的时候，默认情况下会创建6个线程(不包括主进程，主进程负责执行命令)
  - 三个后台进程：异步处理文件关闭任务、AOF刷盘任务、释放内存任务
  - 三个I/O进程
## Redis之前为什么不使用多线程？
- Redis其实在4.0就**引入了多线程进行大键值对数据的删除操作，但是这仅限于后台数据的删除对象，以及阻止通过Redis模块实现的命令**。
- 整体上还是单线程
- 单线程编程容易并且容易维护
- Redis性能瓶颈不在CPU 而在内存和网络
- 多线程存在死锁，线程上下文切换等问题，影响性能
## Redis为什么后面又引入了多线程？
- 6.0版本之后引入多线程也是为了**提高网络IO读写性能**，只是在网络数据的读取这类耗时操作上使用，执行命令仍然还是单线程，不需要担心线程安全问题
# Redis内存管理
## Redis给缓存数据设置过期时间有啥用？
- 因为内存是有限的，如果缓存中的数据都保存的话，分分钟out of memory Redis自带了数据过期的功能。有时候我们业务场景中就是需要某个数据旨在某一个时间段存在，短信验证码，登陆token
## Redis是如何判断数据是否过期的呢？
- 通过一个过期字典(可以看作Hash表)来保存数据过期的时间。过期字典的key指向Redis数据库中的某个Key ，过期字典的值是一个long long类型的整数，这个整数保存了key所指向的数据库键的过期时间
## 过期数据的删除策略？
- 惰性删除: 只会在取出key的时候才对数据进行过期检查，对CPU友好，但是会造成大量的过期key没有被删除
- 定期删除: 每隔一段时间抽取一批key执行删除过期key操作，底层会通过限制删除操作执行的时长和频率来减少操作对CPU时间的影响。
- 定期删除对内存友好，惰性删除对CPU优化，redis是惰性删除+定期删除
## Redis内存淘汰机制
- 从已设置过期时间的数据集中挑选最近最少使用的数据淘汰
- 从已设置过期时间的数据集中挑选将要过期的数据淘汰
- 从已设置过期时间的数据集中任意挑选数据淘汰
- 当内存空间不足以写入新的数据时，在键空间中，移除最近最少使用的key
- 从数据集中任意选择数据进行淘汰
- 当内存不足以写入新数据时，新写入操作报错，不会执行数据添加
- 从已设置过期时间的数据集中挑选最不经常使用的数据淘汰
- 当内存空间不足以写入新的数据时，在键空间中，移除最不经常使用的key
# Redis持久化机制
## 怎么保证Redis挂掉之后再重启数据可以进行恢复？
- 很多时候我们需要持久化数据将内存中的数据写入到硬盘里面，为了重启机器之后还能复用数据
- 持久化方式
  - 快照读 (RDB)
  - 只追加文件(AOF)
## RDB持久化
- Redis可以通过创建快照来获得存储内存里面的数据在某个时间点的副本。Redis创建快照之后，可以对快照进行备份，可以将快照复制到其他服务器从而创建具有相同数据的服务器副本(Redis主从结构)，还可以将快照留在原地用以重启服务器的时候使用
- Redis提供了两个命令来生成RDB快照文件
  - save 主线程执行，会阻塞主线程
  - bgsave 子线程执行 不会阻塞主线程 默认选项
## AOF持久化
- 与快照持久化相比，AOF持久化实时性更好，称为主流的持久化方案
- 开启AOF后每执行一条会更改Redis数据的命令，Redis就会将该命令写入到内存缓存中，再根据appendfsync配置来决定合适将其同步到硬盘中得AOF文件，保存位置和RDBw文件相同，都是通过dir参数设置得，默认文件名是appendonly.aof
- 为了兼具数据和写入性能，appendfsync everysec选项，让Redis每秒同步一次AOF文件，Reids性能几乎不会受影响，即使系统崩溃，也只会丢失一秒之内产生的数据，优雅放慢速度适应硬盘最大的写入速度。
## AOF日志是如何实现的
- 关系型数据可都是执行命令之前记录日志(方便故障恢复)， Redis AOF持久化机制是在执行完命令之后再记录日志 第一步执行命令写入内存，第二步记录日志，写磁盘
  - 避免额外的检查开销，AOF记录日志不会对命令进行语法检查
  - 在执行完之后再记录，不会阻塞当前的命令执行
- 风险：
  - 如果刚执行完命令Redis就宕机会丢失对应的修改
  - 可能会阻塞后续其他命令的执行
## AOF重写
- 当AOF文件变得太大时，Redis会在后台自动重写AOF产生一个新的AOF文件，这个新的AOF文件和原有的AOF文件所保存的数据库状态是一样的，但体积更小
- 无须对现有的AOF文件进行任何读入，分析或者写入操作
- Redis服务器维护一个AOF重写缓冲区，该缓冲区会在子进程创建AOF文件时期，记录服务器执行的所有写命令。当子进程创建完毕之后，服务器会将重写缓冲区中的所有内容追加到新的AOF文件的末尾。新的AOF会替换旧的AOF文件
## 如何选择RDB AOF?
- RDB文件存储的内容是经过压缩的二进制数据，保存着某个时间点的数据集，文件很小，适合做数据的备份和容灾恢复。而AOF文件时存储的每一次写命令，类似于MySQL的binlog文件，会比RDB文件大得多
- 使用RDB文件恢复数据，直接解析还原数据即可，不需要一条一条地执行命令，速度非常快。AOF则需要每次执行命令，速度非常慢。当恢复大数据集时，RDB很快
- RDB数据安全性不如AOF 没有办法实时或者秒级持久化数据，生成RDB文件比较繁琐，AOF文件支持秒级持久化数据，操作轻量
- RDB文件以特定二进制文件保存，Redis演进过程中保存多个版本的RDB，存在低版本不兼容的情况
- AOF文件以一种易于理解和解析的格式包含所有操作的日志，轻松导出AOF文件进行分析，也可以直接操作AOF文件来解决问题。
## Redis4.0对于持久化机制做了什么优化？
- 支持RDB AOF混合持久化
- AOF重写的时候就直接把RDB的内容写入到AOF文件开头，主线程处理的操作命令会被记录才重写缓冲区里，重写缓冲区的增量命令会以AOF方式写入到AOF文件。同时结合了两者的优点，快速加载的同时避免丢失过多的数据。AOF文件中的RDB部分内容是压缩格式不再是AOF文件格式，可读性较差
# Redis事务
  Redis事务是不满足原子性的，也没有办法做到持久性，Redis是不支持回滚的，Redis事务提供了一种将多个命令请求打包的功能，然后再按顺序执行打包的所有命令，并且不会被中途打断。利用Lua脚本来执行多条Redis命令，但是仍然存在问题。Lua脚本中途出现错误退出，出错之后的命令不会执行，但是出错之前已经执行的命令是无法被撤销的，必须保证Lua脚本的所有命令都是正确的。
# Redis性能优化
## bigkey
- 如果一个key对应的value作占用的内存比较大，这个key就可以看做为bigkey，String类型value超过10kb 复合类型的value包含的元素超过5000个，除了会消耗内存还会对性能造成很大影响。
- 使用--bigkeys来查找bigkey或者分析RDB文件来找出bigkey
## 大量key集中过期问题
- 对于key过期问题的处理，采用的是定期删除+惰性/懒汉式删除策略
- 定期删除执行过程中，如果遇到大量的过期key时，客户端请求必须等待定期清理过期key任务的主线程任务执行完毕，这个定期清理key的任务线程是在redis的主线程中执行，此时客户端请求无法被及时清理，相应速度比较慢
  - 给key设置随机过期时间
  - 开启惰性删除/延迟释放。让Redis采用异步方式释放key使用的内存，将操作交给单独的子线程进行处理，避免阻塞主线程
## 使用批量操作减少网络传输
- 命令发送和返回结果耗费时间之和为RTT(往返时间之和)，通过使用批量操作可以减少网路传输次数，进而有效减少网络开销，降低RTT
- Redis存在自己的原生批量操作命令，但是在Redis Cluster下，这些命令会存在一定的问题，mget无法保证所有的Key都在同一个hash slot哈希槽上，mget还是需要多次操作，原子操作也无法保证了。
- Redis Cluster并没有使用一致性哈希，采用的是哈希槽分区。每一个键值对都属于一个哈希槽。当客户端发送命令的时候，会根据对应的公式来计算出key所对应的哈希槽，查询哈希槽和节点的映射关系，即可找到对应的Redis节点
- 整个步骤:
  - 找到key对应的哈希槽
  - 根据哈希槽和节点的映射关系，找到对应的节点，使用mget命令来请求要查询的数据
  - 等待所有命令执行完毕，重装结果数据，保持跟入参key的顺序一致，然后返回结果。
## pipeline
- 对于不支持批量操作的命令，可以利用pipeline(流水线)将命令封装成一组，一次性提交到服务器，只需要一次网络传输，无法保证所有的key都在同一个哈希槽上
- 原生批量操作命令是原子操作，pipeline是非原子操作
- 原生批量操作命令是Redis服务端实现的，pinpeline是客户端和服务端共同实现的
- pipeline可以打包不同的命令，原生操作不可以
- pipeline不适用于执行顺序有依赖关系的一批命令
## Lua脚本
- 无法保证所有的key都在同一个哈希槽上，在Redis Cluster中
- 一段Lua脚本可以看作一条命令执行，可以看作原子操作，Lua脚本的执行过程中不会有其他脚本或者Redis命令同时执行，保证Lua脚本的执行不会被其他命令插入或者打扰。
# Redis生产问题
## Redis缓存穿透
- 大量请求的key是不合理的，既不存在于Redis缓存中，也不存在于数据库中，那么致谢请求就直接到达了数据库上，没有经过缓存这一层，对数据库造成巨大压力，甚至导致数据库宕机。
- 解决办法:
  - 首先进行参数校验
  - 缓存无效的key，不能从根本上解决问题
  - 使用布隆过滤器，把所有可能存在的请求的值都放在布隆过滤器中，当用户请求过来，先判断用户请求的值是否存在于布隆过滤器中，如果不存在直接返回请求参数错误信息给客户端，如果存在就对缓存进行判断，缓存中存在则直接返回数据，缓存中不存在就访问数据库，数据库中存在数据则更新缓存，并将数据返回给客户端，如果数据库中也不存在要查询的数据，直接返回null，空。
### 布隆过滤器
- 二进制向量(位数组)和一系列随即映射函数(哈希函数)两部分组成，占用空间少效率高，但是缺点是返回的结果是概率性的，而不是非常准确的，理论情况下添加到集合中的元素越多，误报的可能性就越大，存放在布隆过滤器中的元素不易被删除
- 当一个元素被加入布隆过滤器的时候:
  - 使用布隆过滤器的哈希函数对元素值进行计算，得到哈希值(存在几个哈希函数就有几个哈希值)
  - 根据得到的哈希值，将对应的位数组的下标设置为1
- 当查询一个元素是否在布隆过滤器的时候:
  - 仍然使用布隆过滤器的哈希函数对元素值进行计算，得到对应的哈希值
  - 得到值之后然后判断位数组中的每个值对应的下标是否为1，如果值全为1，说明这个元素在布隆过滤器中，如果存在一个下表不为1，则说明元素不在布隆过滤器中
- 不同的字符串可能计算出来的哈希位置是一样的，我们应当适当调整我们的哈希函数或者增大位数组的大小，布隆过滤器说存在某个元素是有小概率发生误判的，但是它说不存在某个元素就一定不存在某一个函数
## 缓存击穿
- 请求的key对应的是热点数据，该数据存在于数据库中而不存在于缓存中(缓存中的数据已经过期了)，导致大量的数据请求直接抵达数据库，对数据造成巨大压力
- 解决办法:
  - 设置热点数据永不过期或者过期时间比较长
  - 针对热点数据提前预热，设置合理的过期时间
  - 请求数据库写数据到缓存之前，先获取互斥锁，保证只有一个请求会落在数据库上，减少数据库的压力
- 与缓存穿透的区别: 缓存穿透请求的数据既不存在缓存中，也不存在于数据库中，缓存击穿请求的key对应的是热点数据，数据库中存在该数据，缓存中是不存在该数据的(缓存的数据已经过期)
## 缓存雪崩
- 缓存在一段时间内大面积的失效和瘫痪，导致数据请求直接抵达数据库，数据库承载巨大的压力，甚至可能导致数据库宕机
- Redis服务不可用情况:
  - 构建Redis集群，避免出现单机Redis宕机导致Redis服务不可用
  - 限流，避免太多的数据请求
- 热点数据缓存失效的情况:
  - 设置二级缓存
  - 设置永不过期(不推荐)
  - 设置不同的失效时间比如随机设置失效时间
- 与缓存击穿的区别: 缓存击穿是热点数据过期了或者数据不存在于缓存中了(缓存中的数据过期)，缓存失效，缓存雪崩是同一时间端缓存中大量数据不可用，数据失效了，可能是Redis缓存服务宕机了。
# 如何保证缓存和数据库的数据一致性
- 先更新数据库，再更新缓存和先更新缓存，再删除数据库存在数据不一致性
- 先删除缓存再更新数据库也存在数据不一致性，解决办法是采用延迟双删
- 先更新数据库再删除缓存更合适，但同时为了保证该策略的两个操作都能成功，可以使用消息队列来重试缓存的删除或者订阅Mysql binlog再操作缓存，这两种方法都有一个共同的点是异步操作缓存

# 主从复制
- 是指将一台Redis服务器的数据，复制到其他的Redis服务器，前者称为主节点，后者称为从节点。
- 作用
  - 数据冗余，实现了数据的热备份
  - 故障恢复
  - 负载均衡 主节点提供写服务，从节点提供读服务
  - 高可用基石 哨兵和集群能够实施的基石
- Redis主从的几种常见的拓扑结构
  - 一主一从
  - 一主多从
  - 树状主从
- 主从复制的原理
  - 保存主节点信息
  - 主从建立连接
  - 丛节点发送ping请求进行通信，检查网络是否可用
  - 权限验证
  - 同步数据集
  - 命令持续复制
- 主从数据同步的方式
  - 全量复制：把主节点全部数据一次性发送给从节点
  - 部分复制
- 主从复制存在的问题
  - 主节点的写能力和存储能力收到单机的限制
  - 一旦主节点挂掉，需要手动将一个从节点晋升为主节点，同时需要修改应用方的主节点地址，还需要命令其他从节点去复制新的主节点，全是人为操作
- 哨兵



Redis使用场景
- 缓存  穿透 击穿 雪崩  双写一致 持久化 数据过期和淘汰策略
- 分布式锁  setnx redisson
- 计数器
- 保存token
- 消息队列
- 延迟队列
- 集群问题：主从 哨兵 集群
- 事务：
- redis为什么快

#### 缓存穿透
#### 双写一致
- 业务背景：
  - 一致性要求高
    - 当修改了数据库的数据也要同时更新缓存的数据，缓存和数据库的数据要保持一致
    - 读操作：缓存命中，直接返回；缓存未命中查询数据库，写入缓存，设定超时时间
    - 写操作：延迟双删 延迟的时间并不太好确定
      - 先删除缓存，再操作数据库 会出现脏读
      - 先操作数据库，再删除缓存 会出现脏读
      - 两次删除缓存，降低脏数据的出现   延迟能够保证从节点同步主节点的数据，但仍然不能完全解决问题
    - 分布式锁：
      - 一般放入缓存中的数据都是读多写少  
    - 读写锁：保证强一致性，但效率
      - 读数据的时候，添加共享锁 读读不互斥 读写互斥
      - 写数据的时候，添加排他锁 读读，读写均互斥
  - 允许延迟一致
    - 异步通知保证数据的最终一致性 通过MQ中间件来发布消息并更新缓存 ，更新数据之后，通知缓存删除
#### 持久化
- RDB
  - Redis Database Backup file 数据库备份文件，也称为Redis数据快照，把内存中的所有数据都记录到磁盘中，当Redis实例故障重启后，从磁盘读取快照文件，恢复数据
  - save 主进程执行RDB会阻塞所有命令
  - bgsave 开启子进程执行RDB不会阻塞命令
  - 执行原理： bgsave开始时会fork主进程得到子进程，子进程共享主进程的内存数据，完成fork后读取内存数据并写入RDB文件 拿到记录虚拟地址和物理地址的映射关系即页表
    - fork 采用的copy-on-write技术：主进程执行读操作时，直接访问共享内存    主进程执行写操作时，则会拷贝一份数据，执行写操作 
- AOF
  - Append Only file 追加文件，Redis处理的每一个写命令都会记录再AOF文件，可以看作命令日志文件
  - 因为是记录命令，AOF文件会比RDB文件大得多，AOF会记录对一个key的多次写操作，但只有最后一次写操作才有意义，通过bgrewriteaof命令，可以让AOF文件执行重写公民，以最少命令达到相同效果

#### Redis删除策略  
- 惰性删除和定期删除两种策略进行配合使用
- 惰性删除：设置该key过期时间后，不需要去管他，当需要该key时，我们再检查其是否过期，如果过期，我们将删掉他，反之返回该key
- 定期删除：每隔一段时间，我们就对一些key进行检查，删除里面过期的key （从一定数量的数据库中取出一定数量的随机key进行检查，并删除其中的过期key）同时增加定期删除循环流程的时间上限，**超出时间上限则结束删除**
#### Redis淘汰策略
- 当Redis中的内存不够用时，此时向Redis中添加新的key 那么Redis就会按照某一种规则将内存中的数据删除掉，这种数据的删除规则被称之为内存的淘汰策略
- 共有8种淘汰策略
  - 优先使用allkeys-lru策略 把最近最常访问的数据留在缓存中
  - 如果业务中数据访问频率差别不大，没有明显冷热数据区分，建议使用allkeys-random， 随机选择淘汰
  - 业务中有置顶的需求，使用volatile-lru 同时置顶数据不设置过期时间，这些数据就一直不会被删除，会淘汰其他设置过期时间的数据
  - 业务中有短时高频访问的数据 使用allkeys-lfu 或 volatile-lfu 策略
  - 数据库种有2000万数据，redis只能缓存20w数据，如何保证Redis种的数据都是热点数据？
    - 使用allkeys-lru 淘汰策略，留下来的都是经常访问的热点数据
  - redis内存用完了会发生什么？-根据淘汰策略来决定，默认是不删除任何数据，内存不足直接报错

#### 分布式锁
- 集群情况下的定时任务，抢单，幂等性场景
- 解决跨机器的进程之间的数据同步问题
  - 基于Redis缓冲实现
  - Zookeeper实现
  - 数据库层面实现，乐观锁，悲观锁
- 当使用redisson实现的分布式锁，底层是setnx和lua脚本（保证原子性）
- Redisson实现分布式锁如何合理的控制锁的有效时长？
  - 在redisson的分布式锁种，提供了一个WatchDog(看门狗)，一个线程获取锁成功以后，WatchDog会给持有锁的线程续期(默认是每隔10秒续期一次)
- Redis这个锁可以重入吗？
  - 可以重入，多个所重入需要判断是否是当前线程，在redis种进行存储的时候使用的hash结构来存储线程信息和重入的次数
- Redisson锁能解决主从数据一致的问题吗
  - 不能解决，可以使用Redisson的红锁来解决，但是效率较低，如果要保证强一致性，建议采用zookeeper实现分布式锁
- Zookeeper实现分布式锁
  - 核心思想：当客户端要获取锁，则创建节点，使用完锁，则删除节点
  - 1.客户端获取锁时，在lock节点下创建临时顺序节点 ，临时节点是为了保证服务器宕机了，可以保证节点能被自动删除，防止锁不被释放   顺序节点是为了寻找最小节点，保证节点监听的有序性
  - 2.然后获取lock下面的所有子节点，客户端获取所有的子节点，如果发现自己创建的子节点序号最小，那么就仍未该客户端获取到了锁，使用完了锁将该节点删除
  - 3.如果发现自己创建的子节点并非lock所有子节点序号最小的，说明自己还没有获取到锁，此时客户端需要找到比自己小的那个节点，同时对其注册事件监听器，监听删除事件
  - 4.如果发现比自己小的那个节点被删除，则客户端的Watcher会收到相应通知，此时再次判断自己创建的节点是否是lock子节点种序号最小的，如果是则获取到了锁，如果不是则重复以上步骤继续获取到比自己小的那一个节点并注册监听
  - Zookeeper集群：
    - leader选举
      - serverid:服务器ID 服务器ID编号越大再选择算法种的权重越大
      - Zxid:数据ID 服务器中存放的最大数据ID 值越大说明数据越新，在选举算法种数据越新权重越大
      - 在leader选举中，如果某台Zookeeper获得了超过半数的投票，则可以成为leader
    - 可运行的机器总数没有超过集群中机器的半数，整个集群就将是不会运行的
    - 当集群中的leader宕机了，集群会从剩下的节点中选取一个节点作为leader
  - leader领导:  读写分离  主从复制
    - 处理事务请求 增删改
    - 集群内部各服务器的调度者
  - follower跟随者
    - 处理客户端非事务请求，转发事务请求给leader服务器
    - 参与leader选举投票
  - observer观察者
    - 处理客户端非事务请求，转发事务请求给leader服务器，与follower的区别是不用参与leader选举投票
    - 用来分担follower的客户端非事务请求的压力
#### Redis主从同步
#### 哨兵机制
- 一主一从+哨兵就可以了，单节点内存不超过10G内存
- 实现Redis高并发高可用 实现主从集群的自动故障恢复
- 监控：监控主节点和从节点的状态
  - 基于心跳机制监测服务状态，每隔一秒向集群的每个实例发送ping命令
  - 主观下线：sentinel节点发现某实例未在规定时间响应，则认为该节点已经下线
  - 若超过指定数量的sentinel认为该实例主观下线，则该实例客观下线，数量最好超过sentinel总数量的一半
- 自动故障恢复：主节点发生故障，sentinel将一个从节点会被提升为主节点
- 通知：当集群发生故障时，会将最新的信息推送给redis客户端
- 脑裂
  - 主节点 从节点和sentinel处于不同的网络分区，是的sentinel没有能够心跳感知到主节点，通过选举方式提升一个从节点为主，这样就存在两个master，如同大脑分裂，会导致客户端还在老的主节点那里写入数据据，新节点无法同步数据，当网络恢复之后，sentinel会将老的主节点降为从节点，这是新的master同步数据，造成大量数据据丢失
  - 设置最少的从节点数量以及缩短主从数据同步的延迟时间，达不到要求就拒绝请求
#### 分片集群
- 主从和哨兵能够解决高并发读和高可用的问题，但没有解决海量数据存储和高并发些的问题
- 分片集群特征：
  - 集群中有多个master，每个master保存不同的数据
  - 每个master都可以有多个从节点
  - master之间通过ping监测彼此健康状态
  - 客户端请求可以访问集群任意节点，最终都会被转发到正确的节点
- 哈希槽
  - 集群中每一个节点负责对应的哈希槽 一共16384个哈希槽 将这些哈希槽分配给不同的实例

#### Redis单线程，但是为什么快
- Redis是纯内存操作，执行速度非常快
- 采用单线程，避免了不必要的线程上下文切换，多线程还要考虑线程安全的问题
- 使用I/O多路复用模型，非阻塞IO
  - Redis的性能瓶颈是网络延迟而不是执行速度，IO多路复用模型主要就是实现了高效的网络请求
- 用户空间和内核空间
  - Linux系统为了提高IO效率，会在用户空间和内核空间都加入缓冲区
  - 写数据时，要把用户缓冲数据拷贝到内核缓冲区，然后写入设备
  - 读数据时，要从设备读取数据到内核缓冲区，然后拷贝到用户缓冲区
- 阻塞IO
  - 等待数据和拷贝数据都是阻塞的
- 非阻塞IO
  - recvfrom操作会立即返回结果而不是阻塞用户进程 数据拷贝阶段还是阻塞的
- IO多路复用
  - 利用单个线程监听多个Socket 并在某个Socket可读 可写时得到通知，从而避免无效的等待，充分利用CPU资源
  - select 
  - poll  通知用户线程有Socket就绪，但是不确定是哪一个Socket，需要用户进行逐个遍历Socket进行遍历， select也是如此
  - epoll 在通知用户进程有Socket就绪的同时，把已经就绪的Socket写入用户空间
- Redis网路模型
  - IO多路复用+事件派发 结合事件的处理器应对多个Socket请求
  - 连接应答处理器
  - 命令回复处理器，在Redis6.0之后，为了更好的性能，使用了多线程来处理回复事件
  - 命令请求处理器，在redis6.0之后，将命令的转换使用了多线程，增加命令转换速度，在命令执行的时候，依然是单线程

## RedisObject
- redis的每种对象其实都是有对象结构与对应编码的数据结构组合而成，每种对象类型对应若干编码方式，不同的编码方式对应的底层结构也是不同的
- 对象结构包含的成员变量
  - type 标识该对象是什么类型的对象  String List Set Hash Zset对象
  - encoding 标识该对象使用了哪种底层的数据结构
  - ptr 指向底层数据结构的指针

- SDS
  - 在C语言中，除了字符串的末尾之外，字符串里不能含有"\0"字符，该字符用来判断字符串的长度，否则会被误认为是字符串的结尾。因此C语言的字符串只能用来保存文本数据。C语言的字符串长度获取需要通过遍历方式来统计字符串长度
  - SDS的成员变量
    - len 记录了字符串长度  O(1)复杂度获取字符串长度
    - alloc 分配给字符数组的空间长度     不会发生缓冲区溢出
    - flags 用来标识不同类型的SDS       节省内存空间
    - buf[] 字符数组 用来保存实际数据    二进制安全 
- 链表
- 压缩列表
  - 被设计成一种内存紧凑型的数据结构，占用一块连续的内存空间
- 哈希表
  - 能以O(1)复杂度获取元素，但是插入删除的复杂度是O(n)
  - 哈希冲突
    - 哈希冲突指的是多个键值对通过哈希算法得出的哈希值相同
    - 哈希冲突的解决方法
      - 链地址法
      - 开放地址法
- 整数集合
  - 整数集合是集合键的底层实现之一
  - 整数集合的成员是整数，并且这些整数的取值范围在一个范围内
  - 整数集合的底层实现是数组
  - 整数集合的优点
    - 整数集合的底层实现是数组，因此支持随机访问，时间复杂度是O(1)
    - 整数集合的成员是整数，因此对整数集合进行范围查找的时间复杂度是O(1)
    - 整数集合的成员是整数，因此可以方便地使用整数集合存储整数集合
- **跳跃表**
  - 一种有序的数据结构，它通过在每个节点中维持多个指向其他节点的指针，从而达到快速访问节点的
  - 跳表的优势
    - 插入操作的时间复杂度是O(logn)
    - 节点查找的时间复杂度是O(logn)
  - 跳表每个层级有一个头指针，分别指向了不同层级的节点，每个层级的节点通过指针连接起来
- quicklist
  - 双向链表+压缩列表  它本身是一个链表，而链表中的每个元素又是一个压缩列表 压缩列表存在连锁更新的风险，会造成性能下降
  - 通过控制每个链表节点中的压缩列表的大小或者元素个数，来规避连锁更新的问题。因为压缩列表元素越小或者越少，连锁更新带来的影响就越小。
- listpack
  - 采用连续的内存空间来紧凑地保存数据。 为了节省内存开销，会采用不同的编码方式保存不同大小的数据
  - 他没有像**压缩列表中记录前一个节点长度的字段了**，只会记录当前节点的长度。当我们向listpack加入一个新元素的时候，不会影响其他节点的长度字段的变化，这样就避免了连锁更新的问题。

## redis大key对持久化有什么影响
- 大key对AOF日志的影响
  - 使用Always策略时，主线程在执行fsync()函数时，阻塞的时间会比较长，写入的数据量是很大的时候，这个步骤是很耗时的
  - 使用Everysec策略时，异步执行fsync()，主线程不会阻塞，但是会有一定的丢失数据的风险
  - 使用No策略时，不会执行fsync()函数，主线程不会阻塞，也不会丢失数据，但是数据持久化性能比较差
- 大key对AOF重写和RDB文件的影响
  - 都通过fork()函数创建一个子进程来处理任务，会有两个阶段导致阻塞主进程
    - 第一个阶段是fork()函数执行时，要复制父进程的页表等数据结构，页表越大，阻塞时间越长，会阻塞主线程，但是这个阻塞时间很短，一般只会阻塞0.1秒】
    - 创建完子进程，如果父进程修改了共享数据中的大key，就会发生写时复制，这期间会拷贝物理内存，比较耗时，可能阻塞父进程
- **写时复制**：
  - 写时复制是操作系统实现多进程的一种方式，在父进程和子进程之间，共享数据的时候，如果子进程修改了数据，那么操作系统会为子进程创建一块和父进程共享数据的内存，然后将这块内存的指针赋值给子进程，这样子进程就可以直接操作这块内存，而父进程依然操作自己的内存，这样就避免了父子进程之间同步数据时带来的性能损耗
  - 当父进程或者子进程在向共享内存发起写操作时，CPU会触发写保护终端。这个是由于违反权限导致的，操作系统会在写保护中断处理函数里进行物理内存的复制，并重新设置内存映射关系，将父进程的内存读写权限设置为可读写，最后对内存进行操作。


## 内存淘汰策略 
- 过期删除策略是指代删除已经过期的key，而当redis的运行内存已经超过redis设置的最大内存之后，则会使用内存淘汰策略来删除内存中的key，来保障key的高效运行
- redis的内存淘汰策略
  - 随机删除：随机删除内存中任意一个key
  - 定期删除：每隔一段时间，就随机选取一些key进行删除，但是随机删除和定期删除的策略，都会导致内存中的key不均匀，导致内存淘汰不均匀
  - 内存淘汰策略：
    - noeviction：当内存不足时，不淘汰任何key，直接报错

    - allkeys-lru：当内存不足时，淘汰最近最少使用的key
    - allkeys-lfu：当内存不足时，淘汰使用频率最低的key
    - allkeys-random：当内存不足时，淘汰任意key

    - volatile-lru：当内存不足时，淘汰最近最少使用的key，但是只对设置了过期时间的key生效
    - volatile-lfu：当内存不足时，淘汰使用频率最低的key，但是只对设置了过期时间的key生效
    - volatile-random：当内存不足时，淘汰任意key，但是只对设置了过期时间的key生效
    - volatile-ttl：当内存不足时，淘汰即将过期的key
- lRU算法
  - 最近最少使用算法，淘汰最近最少使用的数据
  - 实现思路：
    - 维护一个双向链表，链表中每个节点表示redis中的一个key，链表中的顺序就是key的访问顺序
    - 每次有新的key加入到缓存中，就将其插入链表头部
    - 当缓存数据被访问时，将数据移到链表头部
    - 链表尾部是最近最少使用的数据
    - 当需要进行内存淘汰时，只需要删除链表尾部的元素即可，因为链表尾部的元素是最久没有使用的
  - 但是redis并没有使用这种方式实现LRU算法，而是使用了一个近似LRU算法的算法，该算法在redis中被称为近似LRU算法
    - 在redis的对象结构种添加一个额外的字段，用来记录此数据的最后一次访问时间。然后使用随机采样的方式来淘汰数据。随机取5个值，然后淘汰最久没有使用的那个。 无法解决缓存污染问题。
- LFU算法，Least Frequently Used，最不经常使用算法
  - 实现思路：
    - 如果数据过去被访问多次，那么将来被访问的频率也更高

## 主从复制
- 尽管AOF和RDB可以完成数据恢复，但是如果数据存储在一台服务器上，一旦服务器出现故障，那么数据将全部丢失，数据恢复还需要一段时间，如果硬盘出现故障，那么数据将无法恢复。
- 为了能够实现数据的高可用，需要将数据复制多份，一旦其中某一台服务器出现故障，可以切换到其他服务器上，这样就可以保证服务器出现故障之后，数据不会全部丢失，同时服务器仍可以对外提供服务
- 主从复制：主从服务器之间是采用读写分离的方式，主服务器负责写，当发生写操作时自动将写操作同步给从服务器，从服务器负责读。**所有数据的修改都在主服务器上进行，主服务器将数据同步给从服务器，从服务器只负责读取数据**
- 通过replicaof 命令，可以将一台服务器设置为另一台服务器的从服务器
- 主从服务器间的第一次同步的过程的三个阶段：
  - 第一阶段：建立连接、协商同步
    - 执行了replicaof命令之后，从服务器就会给主服务器发送psync命令，该命令种包含主服务器的runId和复制进度的offset
    - runId: 每个Redis实例启动时都会随机生成唯一的runId，通过runId可以确定主从服务器的唯一性
    - offset表示复制的进度，第一次同步时，其值为-1
    - 主服务器回复响应，带上两个参数，意图是采用全量复制的方式，也就是主服务器会把所有的数据都同步给从服务器
  - 第二阶段：主服务器同步数据给从服务器
    - 主服务器执行bgsave生成RDB文件并发送给从服务器，从服务器清空当前数据，并把载入RDB文件。但是在该RDB文件可能并没有记录主服务器刚刚生成的写操作数据。为了保证主从一致性，主服务器会将写命令写入到缓冲区里面。
  - 第三阶段：主服务器发送新的写操作命令给服务器
    - 从服务器完成RDB载入后会恢复一个确认消息给主服务器
    - 主服务器再将缓冲区里面的写操作命令发送到从服务器，从服务器再执行完成数据统一
- 基于长连接的命令传播
  - 主从服务器完成第一步同步后会维护一个TCP连接，避免频繁的TCP连接和断开带来的性能开销。
- **增量复制**
  - 需要配置repl_backlog_size，如果他过小，主从服务器网络恢复时，可能发生从服务器想读的数据已经被覆盖了，那么就会导致主服务器采用全量复制的方式。


## 哨兵机制
- 当主服务器挂掉了，主服务器无法完成写操作，需要人工介入，选择一个新的主节点。哨兵机制能够实现主从节点故障转移。
- 监控
  - 哨兵每隔一秒给所有主从节点发送ping命令，主从节点对应地发送一个响应命令给哨兵，以此判断是否正常运行。如果没有在规定时间返回响应，哨兵就会将他们标记为主观下线。
  - 为了减少主观下线带来的误判，通常用多个节点部署成哨兵集群，通过哨兵一起来进行判断。当一个哨兵判断主节点为主观下线后，就会像其他哨兵发起命令，其他哨兵收到该命令之后，就会根据自身和主节点的网络状况，来决定是否同意该命令。
- 选主
  - 一共三轮投票
    - 第一轮投票：判断主节点下线
    - 第二轮投票：选出哨兵leader
    - 第三轮投票：由哨兵leader进行主从故障转移
  - 通过投票进行选出leader来主持主从切换
  - 主从故障转移
    - 选出新主节点
    - 将从节点指向新主节点
    - 通知客户端主节点挂掉了，已经更换了主节点 通过reids的发布者/订阅者机制来实现的
    - 将旧节点变换为从节点2
- 通知
  

## 数据库和缓存如何保证一致性 
- 







#### 雪花算法
- 背景
  - 需要选择合适的方案去应对数据规模的增长，以应对逐渐增长的访问压力和数据量
  - 数据库的扩展方式：业务分库 主从复制 数据库分表
- 数据库分表
  - 垂直分表
  - 水平分表
- 雪花算法能够保证不同表的主键的不重复性，以及相同表的主键的有序性
  - 整体上按照时间自增排序，并且整个分布式系统内不会产生ID碰撞，并且效率较高

#### 逻辑删除
- 物理删除：真实删除，将对应数据从数据库中删除，之后查询不到此条被删除的数据
- 逻辑删除：假删除，将对应数据中代表是否被删除字段的状态改为"被删除状态"





### 互联网项目
- 特点
  - 用户多
  - 流量大，并发高
  - 海量数据
  - 易受攻击
  - 功能繁琐
  - 变更快
- 响应时间： 执行一个请求从开始到最后收到响应数据所花费的总体时间
- 并发数：系统同时能处理的请求数量
  - 并发连接数 客户端向服务器发起请求，并建立了TCP连接，每秒钟服务器连接的总TCP数量
  - 请求数  QPS 每秒多少请求
  - 并发用户数  单位时间有多少用户
- 吞吐量：单位时间内能处理的请求数量
  - QPS 每秒查询数
  - TPS 每秒事务数
  - 一个事务是指客户机向服务器发送请求然后服务器做出反应的过程。客户机在发送请求时开始计时，收到服务器响应后结束计时，以此来计算使用的时间和完成的事务个数
  - QPS >= 并发连接数 >=TPS
- 高性能：提供快速的访问体验
- 高可用:网站服务一直可以正常访问
- 可伸缩：通过硬件增加/减少， 提高/降低处理能力
- 高可扩展： 系统间耦合程度低，方便增加/删除模块
- 安全性：提供网站安全访问和数据加密，安全存储策略
- 敏捷性：随需应变，快速响应
#### 集群
- 集群：很多人一起，干一样的事
- 分布式：很多人一起，干不一样的事儿，这些不一样的事儿，合起来是一件大事
  

### dubbo
- 实现Serializable接口  保证在两个机器之间进行传输  网路传输实现序列化
- dubbo注册中心挂了，服务是否可以正常访问？
  - 可以，dubbo服务消费者在第一次调用时，会将服务提供方地址缓存到本地，以后再调用则不会访问注册中心
- 服务消费者在调用服务提供者的时候发生了阻塞、等待的情形，这个时候，服务消费者会一直等待下去
  - 在某个峰值时刻，大量的请求都在同时请求服务消费者，会造成线程的大量堆积，势必造成雪崩
  - 采用超时机制来解决问题，设置一个超时时间，在这个时间段内，无法完成服务访问，则自动断开连接
  - 设置了超时机制，在这个时间段内无法完成服务访问请求，则自动断开连接
  - 如果出现网络抖动，则这一次请求就会失败，dubbo提供重试机制来避免类似问题的发生，通过retries属性来设置重试次数，默认为2
- 灰度发布：当出现新功能时，会让一部分用户先使用新功能，用户反馈没问题时，再将所有的用户迁移到新功能版本上
  - dubbo中使用version属性来设置调用同一个接口的不同版本
- 负载均衡
  - 策略
    - Random 按权重随机，默认值，按权重设置随机概率
    - RoundRobin  按权重轮询 
    - LeastActive 最少活跃调用数  相同活跃数的随机
    - ConsistentHash  一致性hash 相同参数的请求总是发到同一提供者
- 集群容错
  - 失败重试：当出现失败，重试访问其他服务器，默认重试两次，使用retries配置，一般用于读操作
  - 快速失败：只发起一次调用，失败立即报错，通常用于写操作
  - 失败安全： 出现异常时，直接忽略，返回一个空结果，记录日志
  - 失败自动恢复，后台记录失败请求，定时重发，用于记录重要命令
  - 并行调用多个服务器，只要一个成功即返回
  - 广播调用所有提供者，逐个调用，任意一台报错则报错
- 服务降级
  - mock=force:return null 表示消费方对该服务的方法调用都直接返回null值，不发起远程调用，用来屏蔽不重要服务不可用时对调用方的影响
  - mock=fail:return null 表示消费方对该服务的方法调用在失败后，再返回null值，不抛出异常，用来容忍不重要服务不稳定时对调用方的影响