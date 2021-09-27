library(tidyverse)
library(kevtools)
library(readxl)
library(lubridate)


sourceEnv <- function(path, env) {
  # This will find the most recent file in a directory and source it into its own environment to be accessed
  FILES <- list.files(path, pattern='*.R', full.names=T)
  details <-  file.info(FILES)
  details <- details[order(details$mtime, decreasing=T), ]
  fl <- rownames(details)[1]

  print(paste("sourcing file", fl))
  sys.source(fl, chdir=T, envir=env)
}


CalcCSstate <- function(rcdata) {
  crsr_data <- rcdata %>%
    select(record_id, mrn, test_datetime, crsr_arousal.factor, crsr_auditory.factor, crsr_visual.factor, crsr_motor.factor, crsr_oromotor_verbal.factor, crsr_communication.factor) %>%
    distinct()

  crsr_data[,c(4:9)] <- lapply(crsr_data[,c(4:9)], function(x) as.numeric(as.character(x)))

  consc_states <- crsr_data %>%
    mutate(cs_group = case_when(crsr_arousal.factor == 0 & crsr_auditory.factor < 3 & crsr_visual.factor < 2 & crsr_motor.factor < 3 & crsr_oromotor_verbal.factor <= 1 & crsr_communication.factor == 0 ~ "Coma",
                                crsr_arousal.factor > 0 & crsr_auditory.factor < 3 & crsr_visual.factor < 2 & crsr_motor.factor < 3 & crsr_oromotor_verbal.factor < 3 & crsr_communication.factor < 1 ~ "VS",
                                (crsr_visual.factor > 1 | (crsr_motor.factor > 2 & crsr_motor.factor < 6)) & (crsr_auditory.factor < 3 & crsr_oromotor_verbal.factor < 3 & crsr_communication.factor < 1) ~ "MCSm",
                                (crsr_auditory.factor >= 3 | crsr_oromotor_verbal.factor == 3 | crsr_communication.factor == 1) & (crsr_communication.factor < 2 & crsr_motor.factor < 6) ~ "MCSp",
                                crsr_communication.factor == 2 | crsr_motor.factor == 6 ~ "CS"),
           cs_group = factor(cs_group, levels=c("Coma", "VS", "MCSm", "MCSp", "CS")),
           test_date = as.Date(as.character(test_datetime))) %>%
    arrange(mrn, test_datetime)

  return(consc_states)
}

# All recordings should have a crs-r on the date of the recording regardless.
# so this function is technically not necessary
fillMissingGroup <- function(x) {
  x2 <- x %>%
  group_by(mrn) %>%
    mutate(cs_group = as.character(cs_group)) %>%
  # mutate(cs_group = case_when( (!all(is.na(cs_group)) & any(is.na(cs_group)) ) |
  #                               ( is.na(cs_group) &
  #                             #lag(test_date) + lubridate::days(1)
  #                             # data.table::between(lag(test_date), lower=test_date-days(200), upper=test_date+days(200)) ) ~ lag(cs_group) ,
  #                                 test_date-days(200) <= lag(test_date) & lag(test_date) <= test_date+days(200) ) ~ lag(cs_group),
  # TRUE ~ cs_group))
  mutate(cs_group = case_when(
    all(is.na(cs_group)) ~ NA_character_,
    all(!is.na(cs_group)) ~ cs_group,

    #is.na(cs_group) & test_date-days(200) <= lag(test_date) & lag(test_date) <= test_date+days(200) ~ lag(cs_group),

    TRUE ~ cs_group
  ))


  if (length(which(is.na(x$cs_group))) > length(which(is.na(x2$cs_group)))) {
    fillMissingGroup(x2)

  } else if (length(which(is.na(x$cs_group))) == length(which(is.na(x2$cs_group)))) {
    return(x2)
  }
}


renameMatchCols <- function(db, rn_list) {
  col_idx_to_rn <- which(names(db) %in% unlist(rn_list))
  if (length(col_idx_to_rn) == 0) {
    print("No columns to rename")
    return(db)
  }

  to_rename <- names(db)[col_idx_to_rn]
  new_name_order_idx <- sapply(to_rename, function(x) which(unlist(rn_list) == x))
  new_names <- names(rn_list)[new_name_order_idx]
  names(db)[col_idx_to_rn] <- new_names

  return(db)
}

# this is a better version of base::commandArgs which allows for default arguments to be specified
args <- R.utils::commandArgs(defaults=list(model_output="./psd_out_all.csv",  # "/Volumes/NeurocriticalCare/EEGData/Auditory/cmd_outfiles/psd_out_all.csv",
                                           rc_id="/Volumes/groups/NICU/Consciousness Database/CONSCIOUSNESS_DB_MRN_TO_RECORD_ID.xlsx",
                                           rc_out_path="/Volumes/kd2630/Data/redcap outputs/consciousness/",
                                           save_path="."),
                             asValues=TRUE)

model_output <- args$model_output
rc_id <- args$rc_id
rc_out_path <- args$rc_out_path
save_path <- args$save_path

save_nm <- unlist(lapply(strsplit(model_output, '/'), tail, 1L))
save_nm <- substr(save_nm, 1, nchar(save_nm)-4)
save_nm <- paste0(save_nm, "_w_crsr_group.csv")

# make these paths arguments too
modout <- read.csv(model_output)
rcids <- read_xlsx(rc_id)
rcids$mrn <- as.numeric(as.character(rcids$mrn))

# load redcap data
consc <- new.env()
sourceEnv(path=rc_out_path, env=consc)
data <- consc$data
data2 <- sjlabelled::remove_all_labels(processREDCapData(data))
# add mrns first here
data2 <- data2 %>%
  left_join(select(rcids, record_id, mrn), by = 'record_id')

# repair identical column names
identical_cols <- list(test_datetime=c('eeg_date', 'test_date'))  # any possible other names for test_datetime can go here
data2 <- renameMatchCols(data2, identical_cols)

# create just the file name and then mrn and test date from that
modout$recname2 <- gsub('-raw.fif', '',
                             unlist(lapply(strsplit(modout$rec_name, '/'), tail, 1L)))
# perform surgery to get mrn and date by splitting by _, taking the first two (mrn and date), then pasting them all back together
tmpname1 <- strsplit(modout$recname2, '_')
tmpname2 <- lapply(tmpname1, head, 2)
tmpname3 <- unlist(lapply(tmpname2, paste, collapse='_'))
modout$recname2 <- tmpname3
modout$test_date <- as.Date(unlist(lapply(strsplit(modout$recname2, '_'), tail, 1L)))
if (!'mrn' %in% names(modout)) {
 modout$mrn <- as.numeric(unlist(lapply(strsplit(modout$recname2, '_'), `[[`, 1)))
}

crsrs <- data2 %>%
  mutate(#record_id = as.numeric(record_id),
         test_date = as.Date(test_datetime)) %>%
  # left_join(select(rcids, record_id, mrn), by = 'record_id') %>%
  unite('recname', mrn, test_date, sep='_', remove=F) %>%
  select(record_id, mrn, test_date, test_datetime, recname, crsr_auditory.factor, crsr_visual.factor, crsr_motor.factor, crsr_oromotor_verbal.factor,
         crsr_communication.factor, crsr_arousal.factor, crsr_total) %>%
  filter(!is.na(mrn)) %>%
  distinct()

cs_groups <- CalcCSstate(crsrs) %>%
  # guess I don't need this?
  # unite('recname_join', mrn, test_date, sep='_', remove=F) %>%
  # because there might be multiple entries on a given date across the two databases,
  # need to fill in any values that might have been calculated from either one
  group_by(mrn, test_date) %>%
  fill(cs_group, .direction='updown') %>%
  ungroup() %>%
  # just fill any missing ones with the next closest one.
  #.... by basically doing the same as above but for just mrn.
  group_by(mrn) %>%
  fill(cs_group, .direction='updown') %>%
  ungroup()

cs_groups_join <- cs_groups %>%
  select(mrn, test_date, cs_group) %>%  # recname_join
  distinct()

# # join the recordings TO the crsr data and then do a fill for closest one if it's like a day before or after I GUESS
# alldates_crsr <- cs_groups_join %>%
#   select(mrn, test_date)
#
# alldates_aud <- modout %>%
#   select(mrn, test_date) %>%
#   distinct()
#
# alldates <- unique(rbind(alldates_crsr, alldates_aud))
#
# alldates_cs_group <- alldates %>%
#   left_join(cs_groups_join, by=c('mrn', 'test_date'))
#
#
# tst2 <- tst %>%
#   group_by(mrn) %>%
#   mutate(cs_group = case_when(is.na(cs_group) &
#                               #lag(test_date) + lubridate::days(1)
#                               data.table::between(lag(test_date), lower=test_date-days(200), upper=test_date+days(200)) ~ lag(cs_group),
#   TRUE ~ cs_group))




# alldates_cs_group_filled <- fillMissingGroup(alldates_cs_group)
#
#
# recording_crsrs_closest <- cs_groups_join %>%
#   left_join(modout, by='recname_join') %>%
#   select(mrn, recname_join) %>%
#   distinct()


# add cs group to the model output and re-save it
modout_cs_group <- modout %>%
  left_join(cs_groups_join, by=c('mrn', 'test_date')) %>%  # 'recname_join',  # don't need recname then?
  # arrange(mrn, test_date) %>%
  select(-recname2) %>%
  distinct() %>%
  group_by(rec_name) %>%
  mutate(cs_group = cs_group[as.numeric(cs_group) == min(as.numeric(cs_group), na.rm=T)]) %>%
  distinct()
modout_cs_group$mrn <- as.character(modout_cs_group$mrn)

# ok, so if there's still no score on a given test date, just take the closest one.
# JK, there should always be one for the same date--send this list to Jan and he'll fill in for the recording dates
# that are missing
# write_csv(modout_cs_group, "/Volumes/kd2630/Experiments/Auditory/Consciousness_Recordings_as_of_2021-01-06_with_CRSR_group.csv", na="")

# save model output with the cs_group appended.
write_csv(modout_cs_group, paste0(save_path, "/", save_nm), na="")
