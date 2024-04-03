
-- Noteworthy: 
-- 1. Table and column names not quoted will be uppercased. Quote them to keep them lower. 
-- In Snowflake, primary keys and unique keys are informational only; 
-- they are not enforced but can be used by the query optimizer to improve performance. 
-- When you define a primary key, you're providing Snowflake with information that can be 
-- used to optimize query plans, even though the uniqueness of primary key values is not enforced.
CREATE TABLE IF NOT EXISTS "contributor_stats" (
    "repo" VARCHAR(255),
    "contributor_nodeid" VARCHAR(255),
    "contributor_name" VARCHAR(255),
    "contributor_username" VARCHAR(255),
    "curved_score" FLOAT,
    "stats_beginning" DATE,
    "stats_ending" DATE,
    "contributor_first_commit_date" DATE,
    "num_workdays" INT,
    "commits" INT,
    "prs" INT,
    "review_comments" INT,
    "changed_lines" INT,
    "avg_pr_duration" FLOAT,
    "avg_code_movement_per_pr" FLOAT,
    "commits_per_day" FLOAT,
    "changed_lines_per_day" FLOAT,
    "prs_per_day" FLOAT,
    "review_comments_per_day" FLOAT,
    "prs_diff_from_mean" FLOAT, 
    "prs_ntile" INT,
    "commits_ntile" INT,
    "lines_of_code_ntile" INT,
    "review_comments_ntile" INT,
    "avg_pr_duration_ntile" INT,
    "avg_ntile" INT,
    "median_pr_review_duration" FLOAT,
    PRIMARY KEY ("contributor_nodeid", "repo", "stats_beginning")
);


-- This table is used to stage new ghstats storage
-- since the snowflake library we are using doesn't 
-- support upserts. We store the entire dataframe 
-- here, then use a merge operation to copy it into 
-- the contributor_stats table
CREATE TABLE  IF NOT EXISTS "contributor_stats_staging" (
    "repo" VARCHAR(255),
    "contributor_nodeid" VARCHAR(255),
    "contributor_name" VARCHAR(255),
    "contributor_username" VARCHAR(255),
    "curved_score" FLOAT,
    "stats_beginning" DATE,
    "stats_ending" DATE,
    "contributor_first_commit_date" DATE,
    "num_workdays" INT,
    "commits" INT,
    "prs" INT,
    "review_comments" INT,
    "changed_lines" INT,
    "avg_pr_duration" FLOAT,
    "avg_code_movement_per_pr" FLOAT,
    "commits_per_day" FLOAT,
    "changed_lines_per_day" FLOAT,
    "prs_per_day" FLOAT,
    "review_comments_per_day" FLOAT,
    "prs_diff_from_mean" FLOAT, 
    "prs_ntile" INT,
    "commits_ntile" INT,
    "lines_of_code_ntile" INT,
    "review_comments_ntile" INT,
    "avg_pr_duration_ntile" INT,
    "avg_ntile" INT,
    "median_pr_review_duration" FLOAT,    
    PRIMARY KEY ("contributor_nodeid", "repo", "stats_beginning")
);

CREATE TABLE  IF NOT EXISTS "contributors" (
    "contributor_nodeid" VARCHAR(255),
    "contributor_name" VARCHAR(255),
    "contributor_username" VARCHAR(255),
    PRIMARY KEY ("contributor_nodeid")
);

-- repo_name - the name of the repository
-- stats_beginning - the beginning of the stats period
-- stats_ending - the end of the stats period
-- num_workdays - the number of workdays in the period
-- num_contributors - the number of contributors to the repo
-- avg_pr_duration - the average duration of a pull request across the repo during the period
-- median_pr_duration - the median duration of a pull request across the repo during the period
-- num_prs - the number of pull requests across the repo during the period
-- num_commits - the number of commits across the repo during the period
CREATE TABLE  IF NOT EXISTS "repo_stats" (
    "repo_name" VARCHAR(255),
    "stats_beginning" DATE,
    "stats_ending" DATE,
    "num_workdays" INT,
    "num_contributors" INT,
    "avg_pr_duration" FLOAT,
    "median_pr_duration" FLOAT,
    "num_prs" INT,
    "num_commits" INT,
    PRIMARY KEY ("repo_name", "stats_beginning")
);


-- A repo can have multiple topics. A topic can belong to multiple repos.  
CREATE TABLE if not EXISTS "repo_topics" (
    "repo_name" VARCHAR(255),
    "repo_topic" VARCHAR(255),
    PRIMARY KEY ("repo_name", "repo_topic") 
);

-- PR review comments stored to do AI-based analysis of them
CREATE TABLE IF NOT EXISTS "pr_review_comments" (
    "comment_id" BIGINT PRIMARY KEY,
    "repo_name" VARCHAR(256),
    "pr_number" VARCHAR(64),
    "user_login" VARCHAR(256),
    "body" TEXT,
    "created_at" TIMESTAMP_NTZ
);

-- Staging table for PR review comments so we can more efficiently merge
CREATE TABLE IF NOT EXISTS "pr_review_comments_staging" (
    "comment_id" BIGINT PRIMARY KEY,
    "repo_name" VARCHAR(256),
    "pr_number" VARCHAR(64),
    "user_login" VARCHAR(256),
    "body" TEXT,
    "created_at" TIMESTAMP_NTZ
);


-- SQL here for convenience to backup the contributor_stats table
-- BEGIN TRANSACTION;

-- -- Empty the destination table
-- DELETE FROM "contributor_stats_backup";

-- -- Copy contents from the source table to the destination table
-- INSERT INTO "contributor_stats_backup"
-- SELECT * FROM "contributor_stats";

-- COMMIT;


