## 动态指定数据源
**配置数据源**

```xml
    <environments default="development">
        <environment id="development">
            <transactionManager type="JDBC" />
            <dataSource type="POOLED">
                <property name="driver" value="your_driver_class" />
                <property name="url" value="jdbc:mysql://localhost:3306/default_db" />
                <property name="username" value="your_username" />
                <property name="password" value="your_password" />
            </dataSource>
        </environment>

        <environment id="customDatabase1">
            <transactionManager type="JDBC" />
            <dataSource type="POOLED">
                <property name="driver" value="your_driver_class" />
                <property name="url" value="jdbc:mysql://localhost:3306/custom_db1" />
                <property name="username" value="your_username" />
                <property name="password" value="your_password" />
            </dataSource>
        </environment>

        <environment id="customDatabase2">
            <transactionManager type="JDBC" />
            <dataSource type="POOLED">
                <property name="driver" value="your_driver_class" />
                <property name="url" value="jdbc:mysql://localhost:3306/custom_db2" />
                <property name="username" value="your_username" />
                <property name="password" value="your_password" />
            </dataSource>
        </environment>
    </environments>
```

**定义 mapper 接口**

```java
    public interface DataMapper {
        List<YourResultType> getDataFromDefaultDb(@Param("year") String year);
        List<YourResultType> getDataFromCustomDb1(@Param("year") String year);
        List<YourResultType> getDataFromCustomDb2(@Param("year") String year);
    }
```

**编写 mapperxml 文件**

```xml
    <select id="getDataFromDefaultDb" resultType="YourResultType" parameterType="String">
        SELECT * FROM default_table_${year}
    </select>

    <select id="getDataFromCustomDb1" resultType="YourResultType" parameterType="String">
        SELECT * FROM custom_table_${year}
    </select>

    <select id="getDataFromCustomDb2" resultType="YourResultType" parameterType="String">
        SELECT * FROM another_custom_table_${year}
    </select>
```

**动态切换数据源进行查询**

```java
    String year = "2023";
    if (/* 判断年份对应的数据源 */) {
        sqlSessionFactory.getConfiguration().setEnvironment("customDatabase1");
        DataMapper mapper = sqlSession.getMapper(DataMapper.class);
        List<YourResultType> data = mapper.getDataFromCustomDb1(year);
        // 处理查询结果
    } else {
        sqlSessionFactory.getConfiguration().setEnvironment("default");
        DataMapper mapper = sqlSession.getMapper(DataMapper.class);
        List<YourResultType> data = mapper.getDataFromDefaultDb(year);
        // 处理查询结果
    }
```

## 动态指定查询的表 联表查询

为了处理上述类似的问题，mybatis plus提供了动态表名处理器接口TableNameHandler，我们只需要实现这个接口，并将这个接口应用配置生效，即可实现动态表名。

需要注意的是：
- 在mybatis plus 3.4版本之前，动态表名处理器接口是ITableNameHandler, 需要配合mybatis plus分页插件一起使用才能生效。我们这里只介绍3.4版本之后的实现方式。
- 在mybatis plus 3.4.3.2 作废该的方式：dynamicTableNameInnerInterceptor.setTableNameHandlerMap(map); 大家如果见到这种方式实现的动态表名，也是过时的实现方法，新版本中该方法已经删除。

**实现TableNameHandler接口**
```java

```

**定义mybatisPlusInterceptor拦截器**
```java

```