create table public.date_time_daily_stats as
select 
	extract(year from to_date(substring(date_time from 1 for 10), 'YYYY-MM-DD')) "year",
	extract(month from to_date(substring(date_time from 1 for 10), 'YYYY-MM-DD')) "month",
	extract(day from to_date(substring(date_time from 1 for 10), 'YYYY-MM-DD')) "day",

	-- dimension A: click/booking
	-- dimension B: train/test
    -----------
	--  Click -
	-----------
	-- dimension A
	count(case when (is_booking = 0) then 1 else NULL end) "count_click",

	---------------
	--  Booking  --
	---------------
	-- dimension A
	count(case when (is_booking = 1) then 1 else NULL end) "count_booking",

	-- dimension A+B
	-- booking train
	count(case when (is_booking = 1 and set = 'train') then 1 else NULL end) "count_booking_train",

	-- booking test
	count(case when (is_booking = 1 and set = 'test') then 1 else NULL end) "count_booking_test"

	
from public.overall
group by 1,2,3 -- group by year, month, day
order by 1,2,3 -- order by year, month, day
;


create table public.srch_ci_daily_stats as
select 
	extract(year from srch_ci) "year",
	extract(month from srch_ci) "month",
	extract(day from srch_ci) "day",

	-- dimension A: click/booking
	-- dimension B: train/test
    -----------
	--  Click -
	-----------
	-- dimension A
	count(case when (is_booking = 0) then 1 else NULL end) "count_click",

	---------------
	--  Booking  --
	---------------
	-- dimension A
	count(case when (is_booking = 1) then 1 else NULL end) "count_booking",

	-- dimension A+B
	-- booking train
	count(case when (is_booking = 1 and set = 'train') then 1 else NULL end) "count_booking_train",

	-- booking test
	count(case when (is_booking = 1 and set = 'test') then 1 else NULL end) "count_booking_test"

	
from public.overall
group by 1,2,3 -- group by year, month, day
order by 1,2,3 -- order by year, month, day
;



create table public.srch_ci_cluster_daily_stats as
select 
	extract(year from srch_ci) "year",
	extract(month from srch_ci) "month",
	extract(day from srch_ci) "day",
	hotel_cluster,

	-- dimension A: click/booking
	-- dimension B: train/test
    -----------
	--  Click -
	-----------
	-- dimension A
	count(case when (is_booking = 0) then 1 else NULL end) "count_click",

	---------------
	--  Booking  --
	---------------
	-- dimension A
	count(case when (is_booking = 1) then 1 else NULL end) "count_booking",

	-- dimension A+B
	-- booking train
	count(case when (is_booking = 1 and set = 'train') then 1 else NULL end) "count_booking_train",

	-- booking test
	count(case when (is_booking = 1 and set = 'test') then 1 else NULL end) "count_booking_test"

	
from public.overall
group by 1,2,3,4 -- group by year, month, day, cluster
order by 1,2,3,4 -- order by year, month, day, cluster
;


create table public.srch_co_daily_stats as
select 
	extract(year from srch_co) "year",
	extract(month from srch_co) "month",
	extract(day from srch_co) "day",

	-- dimension A: click/booking
	-- dimension B: train/test
    -----------
	--  Click -
	-----------
	-- dimension A
	count(case when (is_booking = 0) then 1 else NULL end) "count_click",

	---------------
	--  Booking  --
	---------------
	-- dimension A
	count(case when (is_booking = 1) then 1 else NULL end) "count_booking",

	-- dimension A+B
	-- booking train
	count(case when (is_booking = 1 and set = 'train') then 1 else NULL end) "count_booking_train",

	-- booking test
	count(case when (is_booking = 1 and set = 'test') then 1 else NULL end) "count_booking_test"

	
from public.overall
group by 1,2,3 -- group by year, month, day
order by 1,2,3 -- order by year, month, day
;