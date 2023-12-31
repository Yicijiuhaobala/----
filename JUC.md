### 线程和进程
 - 进程: 程序的一次执行过程，系统运行程序的基本单位，进程是动态的，系统运行一个程序就是一个进程从创建、运行到消亡的过程。
 - 线程: 调度CPU资源的基本单位，一个进程可以产生多个线程，**一个进程的多个线程共享进程的堆和方法区的资源，但每个线程都有自己的程序计数器、虚拟机栈和本地方法栈**。线程切换比进程切换的负担小得多，线程称为轻量级进程。一个java程序的运行是main线程和多个其他线程同时进行
 - 区分: 基本上各进程是独立的，而各线程则不一定，因为同一进程中的线程有可能会互相影响，**线程执行开销小，但不利于资源的管理和保护**，而进程正好相反
 - 程序计数器私有: 为了线程切换后能恢复到正确的执行位置
 - 虚拟机栈和本地方法栈私有: 每个java方法在执行的同时会创建一个栈帧用于存储**局部变量表，操作数栈和常量池引用**等信息。**保证线程中的局部变量不被别的线程访问到，虚拟机栈和本地方法栈是线程私有的**

### 堆和方法区
- 堆和方法区是所有线程共享的资源，堆是进程中最大的一块内存，主要用于存放新创建的对象(几乎所有的对象都在这里分配内存)
- 方法区主要用于存放已被加载的类信息，常量，静态变量，及时编译后的代码等数据
### 并发和并行
 - 并发: 两个及两个以上的作业在同一时间段内执行
 - 并行: 两个及两个以上的作业在同一时刻执行
### 同步和异步
 - 同步: 发出一个调用后，在没有得到结果之前，该调用就不可以返回，一直等待
 - 异步: 调用在发出之后，不用等待结果返回，该调用直接返回
### 使用多线程的原因
    - 线程是程序执行的最小单位，线程间的切换和调度的成本远小于进程。多核CPU意味着多个线程可以同时运行，减少了线程上下文切换的开销。
    - 系统高并发量的要求，多线程并发基础正是开发高并发系统的基础，利用多线程机制大大提高系统整体的并发能力和性能
  - 单核时代
    - 单核时代多线程为了提高单进程利用CPU和IO系统的效率。单线程运行时，线程被IO阻塞则整个线程被阻塞，那么此时只有CPU和IO之间的一个被使用，系统利用率低。多线程能保证一个线程被IO阻塞，其他线程能够使用CPU，提高系统资源的利用效率
  - 多核时代
    - 多核时代时，当存在多个CPU时，单个线程只能使用一个CPU，其他CPU不会被利用。而多线程就能在多个CPU上进行执行。在任务中的多个线程没有资源竞争的情况下，任务执行的效率会有显著性的提高
  - **多线程带来的问题: 并发编程能够提高程序的执行效率提高程序运行速度，但也存在其他问题: 内存泄漏，死锁，线程不安全**

### 线程的生命周期和状态
  - java线程运行的生命周期中存在下面六个状态
    - NEW:初始状态，线程被创建可以调用start()
    - RUNNABLE: 线程被调用了start()等待运行
    - BLOCKED: 线程被阻塞，等待锁释放
    - WAITING: 线程等待被分配时间片，等待其他线程做出相应举动 通知或中断
    - TIME_WAITING: 超时等待，可以提前返回不用再继续等待
    - TERMINATED: 线程结束
  - **线程在生命周期中并不是固定处于某一个状态而是随着代码的执行在不同状态之间切换**
### 线程上下文切换
  - 线程在执行过程中会有自己的运行条件和状态，**当线程从占用CPU状态中退出时，需要保存当前线程的上下文，留待线程下次拥有CPU时间片时还能继续之间的工作，同时还要加载即将运行的线程的上下文**。
  - 线程从占用CPU状态中退出:
    - **主动让出CPU 例如wait() sleep()等**
    - 线程的时间片用完，时间片的存在防止一个线程长时间占用CPU资源导致其他线程无法运行或者进程饿死
    - 线程遭遇IO阻塞 阻塞类型的系统中断 请求IO
    - 线程执行完毕退出
  - 线程切换意味着需要保存当前线程的上下文，留待线程下次占用CPU的时候恢复现场，并加载下一个将要占用CPU的线程上下文。
### 死锁
- 多个线程同时被阻塞，他们中的一个或者多个全部都在等待某个资源被释放。由于线程被无限期阻塞，在不借助外界的力量下，程序不可能正常中止！
- 产生死锁的四个条件
  - 互斥条件：该资源任意时刻都只能被一个线程占用
  - 请求与保持条件：一个线程因请求资源而阻塞时，对已经获得资源保持不放
  - 不剥夺条件：线程已经获得的资源在未使用完之前不能被其他线程强行剥夺，只有自己使用完毕后才能释放资源
  - 循环等待条件：若干线程之间形成一种头尾相连的循环等待资源关系
### sleep() 和 wait()的区别  都会暂停线程的执行
  - sleep()方法没有释放锁 wait()释放了锁
  - wait()用于进程间的通信交互， sleep()通常被用于暂停执行
  - wait()方法调用后不会自己苏醒，需要别的线程调用同一对象的notify()或者notifyAll()方法。sleep()执行完毕后会自己苏醒
  - wait()是Object类的本地方法 sleep()是Thread()类的静态本地方法
  - **sleep()是让当前线程暂停执行，不涉及到对象类，也不需要获取对象锁**
  - **wait()方法是让获得对象锁的线程实现等待，那么需要自动释放当前线程占有的对象锁。每个对象都有对象锁，释放当前线程的对象锁需要操作对应的对象而非当前的线程。**
### run() 和start()方法
  - new一个新的Thread()，调用start()方法会启动一个线程使线程进行就绪状态，当分配到时间片就可以开始执行了，start()会执行线程的相应准备工作，然后自动执行run()中的内容，开始多线程工作
  - 直接执行run()方法，会把run()方法当作main线程下的一个普通的方法进行执行，不会在某个线程里执行它，并不是多线程工作。
### 悲观锁和乐观锁
  - 悲观锁: 总是假设最坏的情况，认为共享资源每次被访问的时候就会出现问题(共享数据被修改)，每次在获取资源操作的时候都会上锁，其他线程想要拿到这个资源就需要被阻塞直到拥有资源的线程完成并释放锁。
    - 每次只有一个线程拥有资源，其他线程阻塞， 悲观锁常用于写比较多的场景，避免频繁失败和重试影响性能
  - 乐观锁: 总是假设最好的情况，认为共享资源每次被访问的时候不会出现问题，线程可以不停的执行，不需要加锁也不会阻塞，只是在提交的时候会去检验资源是否已经被其他线程修改了(利用版本号或者CAS算法)
    - 多用于读比较多的场景，避免频繁加锁影响性能，提高系统吞吐量
    - 实现方法:
      - 版本号机制
      - CAS(Compare and swap)算法
        - 用一个预期值和要更新的变量值进行比较，两值相等的时候才会进行更新
    - ABA问题: 如果一个变量V初次读取的时候是A值，并且在准备赋值的时候它仍然是A值，我们不能直接说明他没有被其他线程修改过，因为存在其他线程把V修改为其他值又修改回来，CAS会把它认为没有被修改过。**解决该问题的思路是在变量前面追加版本号或者时间戳**
    - 循环时间长开销大: CAS经常通过自旋操作来进行重试。不成功就一直循环执行直到成功，对CPU执行带来巨大的开销
    - 只能保证一个共享变量的原子操作: 只对单个变量有效，不能应用于多个变量。可以将多个变量放在一个对象里进行加锁操作来实现操作。或者将多个共享变量合成一个共享变量进行操作。

### 并发编程的三个重要特性
  - 原子性: 一次操作或者多个操作，要么所有的操作全部都得到执行并且不受任何影响而中断，要么都不执行 synchronized Lock和原子类实现原子性 synchronized和各种Lock可以保证任一时刻只有一个线程可以访问改代码块。各类原子类利用CAS(或者volatile和final关键字)保证原子性
  - 可见性: 当一个线程对共享变量进行了修改，其他线程是可以立即看见修改后的最新值，利用synchronized和Lock volatile实现可见性。
  - 有序性: 指令重排序可以保证串行语义一致，但不能保证在多线程间的语义一致。使用volatile关键字可以禁止指令进行重排序优化。指令重排序优化在多线程下会出现问题
### JMM Java内存模型
- 注意java内存模型和java内存区域的区别
- java内存模型是java定义的并发编程的一组规范，
- **除了抽象了线程和主内存之间的关系之外，其还规定了从java源代码到CPU可执行指令的这个转换过程中要遵循哪些和并发相关的原则和规范，其主要目的就是简化多线程编程，增强程序可移植性** 
- 线程之间的共享变量必须存储在主内存中。 在当前的java内存模型中，线程可以把变量保存在本地内存中，而不需要直接在主内存中进行读写。这样就造成了一个线程在主内存中修改了共享变量的值，而另外一个线程还继续使用它在本地内存中的变量值，这样就造成了数据的不一致
- 主内存：所有线程创建的实例对象都存储在主内存中，不管该实例对象是成员变量还是方法中的本地变量(局部变量)
- 本地内存： 每个线程都有一个私有的本地内存来存储共享变量的副本，每个线程只能访问自己的本地内存，不能访问其他线程的本地内存。这是一个抽象出来的概念
### volatile关键字
  - volatile关键字可以保证变量的可见性，如果我们将变量声明为volatile，这就指示JVM,这个变量是共享且不稳定的，每次使用它都要到主存中进行读取。它可以保证数据的可见性，但不能保证数据的原子性，synchronized关键字两者都可以保证
  - **volatile可以保证变量的可见性，还能防止JVM的指令重排序  指令重排序在单线程环境下不会出现问题，但是在多线程环境下会导致出现问题**
  - volatile不能保证对变量的操作是原子性的
### synchronized关键字
  - synchronized主要解决的是多个线程之间访问资源的同步性，可以保证被他修饰的方法或者代码块在任意时刻只能由一个线程执行
  - 使用方法:
    - 修饰实例方法: 给当前对象实例加锁，进入同步代码要获得当前对象实例的锁
    - 修饰静态方法: 给当前类加锁，会作用于类的所有对象实例，进入同步代码前要获得当前类的锁
    - 静态 synchronized 方法和非静态 synchronized 方法之间的调用互斥么？不互斥！如果一个线程 A 调用一个实例对象的非静态 synchronized 方法，而线程 B 需要调用这个实例对象所属类的静态 synchronized 方法，是允许的，**不会发生互斥现象**，因为**访问静态 synchronized 方法占用的锁是当前类的锁，而访问非静态 synchronized 方法占用的锁是当前实例对象锁**
    - 修饰代码块:对括号里指定的类和对象加锁
    - 构造方法不能使用sychronized修饰，因为**构造方法本身就是线程安全的，不存在同步的构造方法一说**
    - **synchronized底层原理**
      - synchronized 同步语句块的实现使用的是 monitorenter 和 monitorexit 指令，其中 monitorenter 指令指向同步代码块的开始位置，monitorexit 指令则指明同步代码块的结束位置。
        - 执行monitorenter时会尝试获取锁，如果锁的计数器为0时，则表示锁可以获取，获取锁之后计数器变为1，对象锁的拥有者线程才可以执行monitorexit指令来释放锁，计数器减为0，锁被释放，其他线程可以获取锁 。如果对象回去锁失败，线程就要一直阻塞，直到锁被另外一个线程释放为止
      - synchronized 修饰的方法并没有 monitorenter 指令和 monitorexit 指令，取得代之的确实是 ACC_SYNCHRONIZED 标识，**该标识指明了该方法是一个同步方法，从而执行相应的同步调用**
### volatile和synchroized的区别
  - volatile是线程同步的轻量级实现，但只能用于变量，synchronized可以修饰方法以及代码块
  - volatile可以保证数据的可见性，不能保证原子性，synchronized都可以
  - volatile用于解决变量在多个线程之间的可见性，synchronized解决的是多个线程之间访问资源的同步性
### 锁的四种状态
  - 无锁状态
  - 偏向锁状态
  - 轻量级锁状态
  - 重量级锁状态
  - 锁会随着竞争的激烈而逐渐升级，锁只能升级但不能降级，这是为了提高获得锁和释放锁的效率
### ReentrantLock
  - 一种实现了Lock接口，是一个可重入且独占式的锁，与synchronized关键字类似，它里面有一个内部类 Sync ，继承AQS，添加锁和操作锁的操作都是在Sync中实现的，他有公平锁和非公平锁两个子类。默认使用非公平锁
  - 公平锁: 锁被释放之后，先申请的线程先得到锁，性能较差。公平锁为了保证时间上的绝对顺序，上下文切换更加频繁
  - 非公平锁: 锁被释放之后，后申请的线程可能会先获取到锁，是随机的或者按照其他优先级排序的。性能更好，但某些线程可能永远无法获取到锁
  - 可重入锁: 递归锁，线程可以再次获取到自己的内部锁。比如一个线程获取到了某个对象的锁，此时这个对象还没有被释放，当其想要再次获取这个对象的锁时是可以获取到的。但不可重入锁，是会造成死锁的。  **JDK提供的所有线程的Lock实现类，包括synchronized关键字锁都是可重入的**
  - synchronized依赖于JVM 而ReentrantLock依赖于API 是JDK层面的
  - ReentrantLock相比于synchronized的高级功能:
    - 等待可中断: 正在等待的线程可以选择放弃等待，改为处理其他事情
    - 可实现公平锁: synchronized只能实现非公平锁，
    - 可以实现选择性通知: synchronized关键字与wait()notify()notifyAll()方法相结合可以实现等待/通知机制，ReentrantLock也可以实现，但需要借助于Condition()接口与newCondition()方法
  - **可中断锁: 获取锁的过程可以被中断，不需要一直等到获取锁之后才能进行其他逻辑的处理  ReentrantLock**
  - **不可中断锁: 一旦线程申请了锁，就只能等到拿到锁才能进行其他的逻辑处理  synchronized**

### ReentrantReadWriteLock
- 它是一个可重入的读写锁，既可以保证多个线程同时读的效率，同时又可以保证写入操作时的线程安全
- 一般锁进行并发控制的规则：读读互斥、读写互斥、写写互斥
- 读写锁进行并发控制的规则：读读不互斥、读写互斥、写写互斥
- 它其实是两把锁，一把是写锁，一把是读锁，读锁是共享锁，写锁是排他锁，支持公平锁和非公平锁，默认使用非公平锁
- **它既可以保证多个线程同时读的效率，又能保证有写入操作时的线程安全。在读多写少的情况下，能够明显提升系统性能**



### 共享锁 独占锁
  - 共享锁: 一把锁可以被多个线程同时获得
  - 独占锁: 一把锁只能被一个线程获得
### 线程持有读锁还能再获取写锁吗
  - 持有读锁不能获取写锁，获取写锁的时候，如果发现当前的读锁被占用，就会马上获取失败，不管读锁是不是被当前线程持有
  - 持有写锁可以获取读锁，获取读锁的时候，如果发现写锁被占用，只有写锁没有被当前线程占用的情况下才会获取失败
### 读锁为什么不能升级为写锁
  - 写锁可以降级为读锁，但是读锁不能升级为写锁。读锁升级为写锁会引起线程的争夺，写锁是独占锁，影响性能。
  - 当两个线程同时想升级写锁，则需要对方释放自己的锁，双方都不释放，可能发生死锁





## 线程池
  - 线程池是管理一系列线程的资源池，提供了一种限制和管理线程资源的方式。
    - 降低资源消耗
    - 提高响应速度
    - 提高线程的可管理性
### Executor框架
  - 通过Executor来启动线程比使用Thread的start()方法更好，更易管理，效率更高，用线程池实现，节约开销，有助于避免this逃逸问题
  - 任务 需要实现Runnable接口和Callable接口
  - 任务的执行 ThreadPoolExecutor ScheduleThreadPoolExecutor 这两个类进行任务执行
  - 异步计算的结果 调用submit()方法会返回一个FutureTask对象  execut 返回Future接口对象
### 线程池的创建
  - 通过ThreadPoolExecutor构造函数进行创建
  - 通过Executor框架的工具类Executors进行创建(不推荐) 会造成OOM
    - FixedThreadPool 返回固定数量线程的线程池，线程数量始终不变，提交新的任务有空闲线程则立即执行，反之将任务添加到任务队列中去，等待线程空间来执行  
    - SingleThreadPool 返回一个线程的线程池。多余任务提交都会保存在任务队列中，等待线程空间按先入先出执行
    - CachedThreadPool 返回可以根据实际情况调整线程数量的线程池。当线程有空闲时直接复用线程执行，线程全在执行，且有新的任务，则创建新线程进行执行任务。线程可以返回线程池复用
    - ScheduledThreadPool 返回一个用来在给定延迟后运行任务或者定时运行任务的线程池
    - FixedThreadPool  SingleThreadPool使用的任务队列很大，任务队列的最大长度为Integer.MAX_VALUE  可能造成大量请求堆积，导致OOM   ScheduledThreadPool也可能堆积大量请求 任务队列的最大长度为Integer.MAX_VALUE
    - CachedThreadPool 允许创建的线程数量最大为Integer.MAX_VALUE   可能造成创建大量线程 OOM
### 线程池常见参数
- corePoolSize 任务队列未达到队列容量时，最大可以同时运行的线程数量
- maximumPoolSize 任务队列中存放的任务达到队列容量时，当前可以同时运行的线程数量变为最大线程数
- workQueue 新任务来的时候先判断当前运行的线程数量是否达到核心线程数，如果达到，新任务就会被存放在队列中
- keepAliveTime 线程中的线程数量大于核心线程数时，没有新任务提交，核心线程外的线程不会立即销毁，而是等待超时时间之后才会被销毁
- unit 时间单位
- threadFactory executor 创建新线程时用到
- handler 饱和策略


### 线程池的阻塞队列 与内置线程池有关
- 容量为Integer.MAX_VALUE的LinkedBlockingQueue(无界队列)  FixedThreadPool  SingleThreadPool。队列永远不会被放满， FixedThreadPool最多只能创建核心线程数的线程
- SynchronousQueue(同步队列) CachedThreadPool 它可以创建无限数量的线程，最大容量为Integer.MAX_VALUE。
- DelayWorkQueue(延迟阻塞队列)  内部的元素并不是按照放入的时间进行排序的，会按照延迟的时间长短对任务进行排序，内部采用的是"堆"的数据结构，可以保证每次出队的任务都是当前队列中执行时间最靠前的。每次扩容为原来容量的1/2，最大容量为Integer.MAX_VALUE，最多只能创建核心线程数的线程


### 线程池的饱和策略
  - ThreadPoolExecutor.AbortPolicy 抛出错误拒绝执行任务
  - ThreadPoolExecutor.CallerRunPolicy 调用执行自己的线程运行任务 直接在调用execute方法的线程中运行被拒绝的任务 执行被关闭，会丢弃该任务。 影响新任务的提交速度，影响系统的整体性能
  - ThreadPoolExecutor.DiscardPolicy 不执行任务，直接丢弃掉该任务
  - ThreadPoolExecutor.DiscardOldestPolicy 丢弃最早的未处理的任务
### 线程池处理任务的流程
  - 当前运行的线程小于核心线程数，新建一个线程来执行任务
  - 当前运行的线程不小于核心线程数，但小于最大线程数，将任务添加到任务队列中
  - 当任务队列已经满了之后，当前运行的线程小于最大线程数，新建一个线程执行任务
  - 当前运行的线程数已经等于最大线程数，不能再新建线程了，当前任务被拒绝，饱和策略
### 线程池命名
  - 初始化线程池的时候需要显式命名(设置线程池名称前缀)，有利于定位
  - 利用guava的ThreadFactoryBuilder
  - 自己实现ThreadFactor
### 设定线程池的大小
  - 多线程增加了上下文切换的成本
  - 线程池太小时，当大量任务进行请求处理时，会导致大量的请求任务在任务队列中等待执行，甚至出现任务队列满了某些任务甚至无法执行和处理，任务堆积再任务队列中导致OOM
  - 线程池太大时，大量线程争夺CPU资源，导致大量的上下文切换，增加线程执行时间，影响系统的整体执行效率
  - CPU密集型任务(N+1) N为CPU核心数
  - I/O密集型任务(2N)
  - 如何判断是CPU密集型任务还是IO密集型任务
    - CPU密集型任务简单理解就是利用CPU计算能力的任务比如你在内存中对大量数据进行排序。
    - 但凡设计到网路读取，文件读取这类都是IO密集型，这类任务的特点是CPU计算耗费时间相比于等待IO操作完成的时间来说很少，大部分时间都花在了等待IO操作完成上。
### 动态修改线程池的参数
  - corePoolSize  核心线程数
  - maximumPoolSize  最大线程数
  - workQueue  任务队列
### Future 
  - 异步思想的典型应用，主要用在一些需要处理耗时任务的场景，避免任务一直原地等待耗时任务执行完毕，执行效率太低。将耗时任务交给子线程去异步执行，自己去处理其他任务，之后再去获取任务执行结果
  - 四个功能
    - 取消任务
    - 判断任务是否被取消
    - 判断任务是否已经执行完毕
    - 获取任务的执行结果
    - 我有一个任务提交给了Future来处理，任务执行期间可以去做任何想做的事情，并且在这个期间还可以取消任务和获取任务的执行状态，也可以在任务执行之后获取任务执行结果
  - Callable Future的关系
    - FutureTask提供了Future接口的基本实现，常用来封装Callable 和 Runnable，具有取消任务、查看任务是否执行完成以及获取任务执行的结果的方法。
  - CompletableFuture类有什么用 
    - Future不支持异步任务的编排组合，获取计算结果的get()方法是阻塞调用。  CompletableFuture提供了函数式编程、异步任务编排组合
### ThreadLocal
  - 通常情况下我们创建的变量是可以被任何一个线程访问并修改的，**ThreadLocal主要解决让每个线程绑定自己的值，保证每个线程拥有自己的私有数据**。
  - 创建一个ThreadLocal变量，那么访问这个变量的每个线程都会有这个变量的本地副本。可以直接使用get() set()方法来获取或者更改数据，避免了线程安全问题。
  - ThreadLocalMap是ThreadLocal类的实现的定制化的HashMap 调用get() set()方法实际上是ThreadLocalMap的方法，可以知道最终的变量是放在了当前线程的ThreadLocalMap上。通过getMap(Thread t)获取线程对象
  - **ThreadLocalMap的key是ThreadLocal对象，value是所设置的值。ThreadLocalMap是ThreadLocal的静态内部类**
  - ThreadLocalMap中的key为弱引用，value是强引用。如果ThreadLocal没有被外部强引用的情况下，一般在垃圾回收中，会出现key被回收，value不会被回收的情况。那么key就会为null,value永远不会被回收，出现内存泄漏。在调用set() get() remove()方法时，清理掉key为null的记录。使用完ThreadLocal后最好手动调用remove()方法
  - 弱引用
    - 如果一个对象只具有弱引用，就类似于可有可无的生活用品。弱引用与软引用的区别在于：**只具有弱引用的对象拥有更短暂的生命周期**。在垃圾回收器线程扫描它所管辖的内存区域的过程中，**一旦发现只具有弱引用的对象，不管内存空间是否足够与否，都会回收它的内存** 垃圾回收器是一个优先级很低的线程，因此不一定很快发现那些只具有弱引用的对象
    - 弱引用可以和一个引用队列联合使用，如果弱引用所引用的对象被垃圾回收，java虚拟机就会把这个弱引用加入到与之关联的引用队列中
  - 强引用
    - new出来的对象就是强引用类型，只要强引用存在，**垃圾回收器将永远不会回收被引用的对象，哪怕内存不足的时候**
  - 软引用
    - 使用SoftReference修饰的对象被称为软引用，**软引用指向的对象在内存要溢出的时候被回收**
  - 虚引用
    - 最弱的引用，作用就是用队列接受对象即将死亡的通知
  - set()方法
    - 主要是判断ThreadLocalMap是否存在，然后使用ThreadLocalMap的set()方法进行数据处理
  - Hash算法
    - int i = key.threadLocalHashCode & (len-1)  i就代表key在散列表中对应数组的下标位置  每当创建一个ThreadLocal对象，key.threadLocalHashCode会增长一个固定的值， 黄金分割数，保证hash均匀分布
  - Hash冲突
    - **HashMap中解决冲突的方法是在数组上构造一个链表结构，冲突的结构挂在到链表上，链表长度超过一定数量会转化成红黑树**
    - 发现槽位存在Entry数据，就会进行线性查找，一直找到Entry为nul的槽位才会停止查找，将当前元素放入此槽位中。

  
### AQS
- AbstractQueueSynchronizer 抽象队列同步器  主要用来构建锁和同步器   使用AQS能简单且高效地构造出应用广泛的大量同步器
- **核心思想：如果被请求的共享资源空闲，则将当前请求资源的线程设置为有效的工作线程，并且将共享资源设置为锁定状态。如果被请求的共享资源被占用，那么就需要一套线程阻塞等待以及被唤醒时锁分配的机制，这个机制是用CLH队列锁来实现的，即将暂时获取不到锁的线程加入到队列中去**
- CLH(Craig,Landin,and Hagersten) 队列是一个虚拟的双向队列（虚拟的双向队列即不存在队列实例，仅存在结点之间的关联关系）。AQS 是将每条请求共享资源的线程封装成一个 CLH 锁队列的一个结点（Node）来实现锁的分配。在 CLH 同步队列中，一个节点表示一个线程，它保存着线程的引用（thread）、 当前节点在队列中的状态（waitStatus）、前驱节点（prev）、后继节点（next）。
- 以 ReentrantLock 为例，state 初始值为 0，表示未锁定状态。A 线程 lock() 时，会调用 tryAcquire() 独占该锁并将 state+1 。此后，其他线程再 tryAcquire() 时就会失败，直到 A 线程 unlock() 到 state=0（即释放锁）为止，其它线程才有机会获取该锁。当然，释放锁之前，A 线程自己是可以重复获取此锁的（state 会累加），这就是可重入的概念。但要注意，获取多少次就要释放多少次，这样才能保证 state 是能回到零态的。
- 再以 CountDownLatch 以例，任务分为 N 个子线程去执行，state 也初始化为 N（注意 N 要与线程个数一致）。这 N 个子线程是并行执行的，每个子线程执行完后countDown() 一次，state 会 CAS(Compare and Swap) 减 1。等到所有子线程都执行完后(即 state=0 )，会 unpark() 主调用线程，然后主调用线程就会从 await() 函数返回，继续后余动作。

### Semaphore 
- Semaphore信号量可以用来控制同时访问特定资源的线程数量
- **假设有N个线程来获取Semaphore中的共享资源，同一时刻中只有N个线程能获取到共享资源，其他线程都会阻塞**。只有获取到共享资源的线程才能执行，等到有线程释放了共享资源，其他阻塞线程才能获取到。   **当初始的资源个数为1的时候，将会退化为排他锁**
- 模式  默认为非公平模式
  - 公平模式 调用acquire()方法的顺序就是获取许可证的顺序 遵循FIFO
  - 非公平模式： 抢占式模式  
- **通常用于那些资源有明确访问数量限制的场景比如限流**，但是仅限于单机， 分布式项目中建议使用redis+lua
- 原理
  - 共享锁的实现，默认构造AQS的state的值为许可证的数量，只有拿到许可证的线程才能执行
  - 调用acquire()，线程尝试获取锁，**如果state>0表示获取成功，同时使用CAS操作去修改state state=state-1  如果state<0的话，则表示许可证数量不足，此时会创建一个Node节点加入阻塞队列，挂起当前线程**
  - 调用semaphore.release(); **线程尝试释放许可证，并使用 CAS 操作去修改 state 的值 state=state+1。**释放许可证成功之后，同时会唤醒同步队列中的一个线程。**被唤醒的线程会重新尝试去修改 state 的值 state=state-1 ，如果 state>=0 则获取令牌成功，否则重新进入阻塞队列，挂起线程**。
  
### CountDownLatch
- 它允许count个线程阻塞在一个地方，直至所有的线程任务都执行完毕，它是一次性的，在构造方法中初始化一次，使用完之后就不能再次被使用了
- CountDownLatch 是共享锁的一种实现,它默认构造 AQS 的 state 值为 count。**当线程使用 countDown() 方法时,其实使用了tryReleaseShared方法以 CAS 的操作来减少 state,直至 state 为 0** 。**当调用 await() 方法的时候，如果 state 不为 0，那就证明任务还没有执行完毕，await() 方法就会一直阻塞，也就是说 await() 方法之后的语句不会被执行。然后，CountDownLatch 会自旋 CAS 判断 state == 0，如果 state == 0 的话，就会释放所有等待的线程，await() 方法之后的语句得到执行**。
- 作用：**允许count个线程阻塞在一个地方，直至所有线程的任务都执行完毕**。使用场景：多线程读取多个文件处理的场景，要获取所有文件处理的结果统计
- 
### CyclicBarrier
- 让一组线程到达一个屏障(也叫同步点)时被阻塞，知道最后一个线程到达屏障时，屏障才会开门，所有被屏障拦截的线程才会继续干活。
- 内部通过一个count变量作为计数器，**count初始值为parties属性的初始化值，每当一个线程到达了栅栏，就将计数器减一**，如果count值为0了，表示这是最后一个线程抵达栅栏，就尝试执行我们构造方法中输入的任务。




## Java常见并发容器
- ConcurrentHashMap
  - 
- CopyOnWriteArrayList
  - 为了将读操作性能发挥到极致，CopyOnWriteArrayList中的读取操作是完全无需加锁的，更加厉害的是写入操作也不会阻塞读取操作，**只有写写才会互斥**，这样一来，读操作的性能就大大提升了
  - 其采用了写时复制(Copy-on-write)策略 当需要修改(add set remove 等操作)CopyOnWriteArrayList的内容时，**不会直接修改原数组，而是会先创建底层数组的副本，对副本数组进行修改，修改完之后再将修改后的数组赋值回去，这样就可以保证写操作不会影响读操作了**
- ConcurrentLinkedQueue
  - 主要使用CAS非阻塞算法来实现线程安全，
  - 阻塞队列可以用过加锁来实现，非阻塞队列可以通过CAS操作来实现
- BlockingQueue
  - 阻塞队列被广泛应用于“生产者-消费者”问题中，它提供了可阻塞的插入和移除的方法。当队列容量满的时候，生产者线程会被阻塞，直到队列未满；当队列容器为空时，消费者线程就会被阻塞，直至队列非空时为止
  - ArrayBlockingQueue
    -  底层采用数组来实现，它一旦被创建，容量就不能改变了，并发控制采用可重入锁ReentrantLock，不管是插入操作还是读取操作，都需要获取到锁才能进行操作。 **默认情况下不能保证线程访问队列的公平性**
    - 公平性：严格按照线程等待的绝对时间顺序，即最先等待的线程能够最先访问到ArrayBlockingQueue
    - 非公平性：访问ArrayBlockingQueue并不是按照线程等待的绝对时间顺序。 如果保证绝对性，需要降低吞吐量。
  - LinkedBlockingQueue
    - 底层基于单向链表实现的阻塞队列，可以做无界队列也可以做有界队列，满足FIFO的特性  通常在创建的时候会指定它的大小，没有指定的情况下容量等于Interger.MAX_VALUE
  - PriorityBlockingQueue 
    - 一个支持优先级的无界阻塞队列 默认情况下采用自然顺序进行排序  也可以自定义类实现compareTo()方法来指定元素排序规则 初始化时通过构造器参数Comparator来指定排序规则
    - 不可以插入null值，插入队列的对象必须是可比较大小的。
    - 当后面插入元素的时候如果空间不够的话会自动扩容
- ConcurrentSkipListMap
  - 对平衡树的插入和删除往往很可能导致平衡树进行一次全局的调整，而对调表的插入和删除只需要对整个数据结构的局部进行操作即可。
  - 需要一个全局锁来保证整个平衡树的线程安全，而对于调表，只需要部分锁就可以了。**跳表的时间复杂度为O(logn)**
  - 跳表是一种利用空间换时间的算法   使用哈希实现Map并不会保存元素的顺序  跳表内所有的元素是有序的。