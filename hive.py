#
# drop table if exists ***;
# create table *** as
# select * from
#
# # 建表语句
# create table ***(
#     col1 string comment '',
#     col2 bigint comment '',
#     col3 double comment ''
# )
# comment '**表'
# partition by (dt string)
# row format delimited fields terminated by '|'
# stored as orc
# tblproperties ('owner'='', 'pformat'='dt:yyyy-mm-dd', 'pt_nd'='730', 'tbl_wnd'='730');
#
# # 增加新列
# alter table ** add columns(col4 string comment '') cascade;
# alter table ** add columns(col5 ARRAY<bigint> comment '') cascade;