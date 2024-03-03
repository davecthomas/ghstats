
-- Noteworthy: 
-- 1. Table and column names not quoted will be uppercased. Quote them to keep them lower. 
-- In Snowflake, primary keys and unique keys are informational only; 
-- they are not enforced but can be used by the query optimizer to improve performance. 
-- When you define a primary key, you're providing Snowflake with information that can be 
-- used to optimize query plans, even though the uniqueness of primary key values is not enforced.
CREATE TABLE "contributor_stats" (
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
    PRIMARY KEY ("contributor_nodeid", "repo", "stats_beginning")
);


-- This table is used to stage new ghstats storage
-- since the snowflake library we are using doesn't 
-- support upserts. We store the entire dataframe 
-- here, then use a merge operation to copy it into 
-- the contributor_stats table
CREATE TABLE "contributor_stats_staging" (
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
    PRIMARY KEY ("contributor_nodeid", "repo", "stats_beginning")
);

CREATE TABLE "contributors" (
    "contributor_nodeid" VARCHAR(255),
    "contributor_name" VARCHAR(255),
    "contributor_username" VARCHAR(255),
    PRIMARY KEY ("contributor_nodeid")
);

CREATE TABLE "repo_stats" (
    "repo_name" VARCHAR(255),
    "stats_beginning" DATE,
    "stats_ending" DATE,
    "avg_pr_duration" FLOAT,
    "median_pr_duration" FLOAT,
    "num_prs" INT,
    "num_commits" INT,
    PRIMARY KEY ("repo_name", "stats_beginning")
);


-- A repo can have multiple topics. A topic can belong to multiple repos.  
CREATE TABLE "repo_topics" (
    "repo_name" VARCHAR(255),
    "repo_topic" VARCHAR(255),
    PRIMARY KEY ("repo_name", "repo_topic") 
);
