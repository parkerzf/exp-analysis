CREATE TABLE public.train
(
	date_time character(21),
	site_name character varying(8),
	posa_continent character varying(8),
	user_location_country character varying(8),
	user_location_region character varying(8),
	user_location_city character varying(8),
	orig_destination_distance double precision,
	user_id character varying(32),
	is_mobile integer,
	is_package integer,
	channel character varying(8),
	srch_ci date,
	srch_co date,
	srch_adults_cnt integer,
	srch_children_cnt integer,
	srch_rm_cnt integer,
	srch_destination_id integer,
	srch_destination_type_id character varying(8),
	is_booking integer,
	cnt integer,
	hotel_continent character varying(8),
	hotel_country character varying(8),
	hotel_market integer,
	hotel_cluster integer
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.train
  OWNER TO "feng.zhao";

# Normalize the schema
ALTER table public.train Add column id character varying(32) NOT NULL DEFAULT '-1';
ALTER table public.train Add column set character varying(10) NOT NULL DEFAULT 'train';



CREATE TABLE public.test
(
	id character varying(32),
	date_time character(21),
	site_name character varying(8),
	posa_continent character varying(8),
	user_location_country character varying(8),
	user_location_region character varying(8),
	user_location_city character varying(8),
	orig_destination_distance double precision,
	user_id character varying(32),
	is_mobile integer,
	is_package integer,
	channel character varying(8),
	srch_ci date,
	srch_co date,
	srch_adults_cnt integer,
	srch_children_cnt integer,
	srch_rm_cnt integer,
	srch_destination_id integer,
	srch_destination_type_id character varying(8),
	hotel_continent character varying(8),
	hotel_country character varying(8),
	hotel_market integer
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.test
  OWNER TO "feng.zhao";

# Normalize the schema
ALTER table public.test Add column set character varying(10) NOT NULL DEFAULT 'test';
ALTER table public.test Add column is_booking integer NOT NULL DEFAULT 1;
ALTER table public.test Add column cnt integer NOT NULL DEFAULT 1;
ALTER table public.test Add column hotel_cluster integer;

# Union to get the overall table
CREATE TABLE public.overall AS (

SELECT id, set, date_time,site_name,posa_continent,user_location_country,user_location_region,
user_location_city,orig_destination_distance,user_id,is_mobile,is_package,channel,srch_ci,
srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,srch_destination_type_id,is_booking,cnt,
hotel_continent,hotel_country,hotel_market,hotel_cluster
FROM public.train

Union

SELECT id, set, date_time,site_name,posa_continent,user_location_country,user_location_region,
user_location_city,orig_destination_distance,user_id,is_mobile,is_package,channel,srch_ci,
srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,srch_destination_type_id,is_booking,cnt,
hotel_continent,hotel_country,hotel_market,hotel_cluster
FROM public.test);

# Add index on is_booking field
CREATE INDEX index_booking ON public.overall (is_booking);



