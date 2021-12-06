import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytz
from context_aware.interfaces.api import ContextAwareAPI
from context_aware.interfaces.api.context_aware_events import EventDumpParser

# from context_aware.interfaces.empatica.obelisk import EmpaticaObeliskClient
from context_aware.interfaces.api.idlab_cloud import RcloneNextcloudSync
from context_aware.interfaces.empatica.obelisk import EmpaticaObeliskv3Client
from context_aware.utils.hooks import MattermostAlertManager
from context_aware.visualizations.plotly import figs_to_html
from functional import seq
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

obelisk_dir = Path(
    "/users/jonvdrdo/jonas/data/aaa_contextaware/raw/mbrain/obelisk_dump/"
)

# patient_id = "patient_Jonas"


def get_user_wear_time_df(patient_id: str, window_days=30) -> pd.DataFrame:
    # de data van elke user querien en op een 00/24u graph via kleine rectangles (v-lines) weergeven wanneer een user de wearable draagt
    today = pd.Timestamp(datetime.now(tz=pytz.timezone("Europe/Brussels")))
    signal = "tmp"
    t_start = pd.Timestamp(today - timedelta(days=window_days))
    eligible_folders = list(obelisk_dir.glob(f"{patient_id}.empatica.*"))
    eligible_files: List[Path] = []
    for eligible_folder in eligible_folders:
        for date in pd.date_range(t_start.date(), today.date(), freq="D"):
            eligible_files.extend(
                eligible_folder.glob(f'{signal}_{date.strftime("%Y_%m_%d")}.parquet')
            )

    data = []
    for eligible_file in tqdm(list(eligible_files)):
        pqt = pd.read_parquet(eligible_file).set_index("timestamp")
        if len(pqt):
            data.append(pqt)

    if len(data):
        df_tmp = pd.concat(data).sort_index()
        del data
    else:
        return pd.DataFrame()

    fs_tmp = 4
    data = []
    for date in tqdm(
        pd.date_range(t_start.date(), today.date(), freq="D").tz_localize(
            "europe/brussels"
        )
    ):
        df_day = df_tmp[date : date + pd.Timedelta(days=1)]
        #         print(date, df_day)

        if not len(df_day):
            # TODO: maybe add empty slot for that day
            continue

        gaps_start = df_day.index.to_series().diff() > timedelta(seconds=1.1 / fs_tmp)
        gaps_start.iloc[[0]] = True

        gaps_end = gaps_start.shift(-1)
        gaps_end.iloc[[-1]] = True

        gaps_start = df_day[gaps_start].index.to_list()
        gaps_end = df_day[gaps_end].index.to_list()
        data += [
            [patient_id, date, gap_start, gap_end]
            for gap_start, gap_end in zip(gaps_start, gaps_end)
        ]

    user_sessions = pd.DataFrame(
        data=data, columns=["patient_id", "date", "start", "end"]
    )
    user_sessions["session_dur"] = user_sessions.end - user_sessions.start

    user_sessions = pd.merge(
        user_sessions,
        user_sessions.groupby("date")["session_dur"].sum().rename("daily_wear_time"),
        left_on="date",
        right_index=True,
        how="outer",
    )
    return user_sessions


daytime_to_0 = dict(minute=0, hour=0, second=0, microsecond=0)

DEBUG = False

# dictionary which is used to aggregated all warnings / erros
mattermost_dict = {"info": {}, "warning": {}, "errors": {}}


# ---------------------------- helper methods ----------------------------
def get_questionnaire_notifications(
    notifications: List[dict], questionnaire_names: List[str]
):
    def _check_name_in_questionnaire(questionnaire_data: Union[Dict, List]):
        if isinstance(questionnaire_data, dict):
            return questionnaire_data.get("name", "") in questionnaire_names
        elif isinstance(questionnaire_data, list):
            return any([q_n in questionnaire_names for q_n in questionnaire_data])
        else:
            print("invalid type")
            print(questionnaire_data)

    out = list(
        seq(notifications)
        .filter(lambda x: x.get("data", {}).get("questionnaires", None) is not None)
        .filter(lambda x: _check_name_in_questionnaire(x["data"]["questionnaires"]))
    )
    return out


# ---------------------------- interaction rate checking ----------------------------
def check_daily_questionnaire_interaction(
    patient_id: str, user_id: str, max_unanswered_questionnaires=5, window_days=30
):
    today = datetime.now(tz=pytz.timezone("Europe/Brussels")).replace(**daytime_to_0)
    t_start = today - timedelta(days=window_days)

    notifs = ContextAwareAPI.get_notification_user_interval(user_id, t_start, today)
    events = ContextAwareAPI.get_events_user_interval(user_id, t_start, today)

    quest_metadata_dict = {}
    for quest_name in ["morning_nl", "evening_nl"]:
        # assumption: the number of notifs is equal to the number of days/questionnaires
        #   that the participant needs to have filled in
        questionnaire_notifs = get_questionnaire_notifications(
            notifs, questionnaire_names=[quest_name]
        )
        questionnaire_events = list(
            seq(events)
            .filter(lambda x: x.get("type", "") == "questionnaire")
            .filter(lambda x: x.get("payload", {}).get("name", "") == quest_name)
        )

        quest_metadata_dict[quest_name] = {
            "notifications": len(questionnaire_notifs),
            "answers": len(questionnaire_events),
        }

        warning_msg = "\n".join(
            [
                f"user: `{patient_id}`; questionnaire **{quest_name}**",
                f"\tstats: _window=`{window_days}days`; "
                f"max_not_answered=`{max_unanswered_questionnaires}`_",
                f"> user answered only {len(questionnaire_events)} questionnaires out "
                f"of {len(questionnaire_notifs)} sent notifications",
            ]
        )

        n_unanswered_questionnaires = len(questionnaire_notifs) - len(
            questionnaire_events
        )


#         if n_unanswered_questionnaires > max_unanswered_questionnaires:
# elif n_unanswered_questionnaires >= max_unanswered_questionnaires // 2:
#     a.warning(module="interaction rate", warning_text=warning_msg)


def get_user_wear_time_df(patient_id: str, window_days=30) -> pd.DataFrame:
    # de data van elke user querien en op een 00/24u graph via kleine rectangles (v-lines) weergeven wanneer een user de wearable draagt
    today = pd.Timestamp(datetime.now(tz=pytz.timezone("Europe/Brussels")))
    signal = "tmp"
    t_start = pd.Timestamp(today - timedelta(days=window_days))
    eligible_folders = list(obelisk_dir.glob(f"{patient_id}.empatica.*"))
    eligible_files: List[Path] = []
    for eligible_folder in eligible_folders:
        for date in pd.date_range(t_start.date(), today.date(), freq="D"):
            eligible_files.extend(
                eligible_folder.glob(f'{signal}_{date.strftime("%Y_%m_%d")}.parquet')
            )

    data = []
    for eligible_file in tqdm(list(eligible_files)):
        pqt = pd.read_parquet(eligible_file).set_index("timestamp")
        if len(pqt):
            data.append(pqt)

    if len(data):
        df_tmp = pd.concat(data).sort_index()
        del data
    else:
        return pd.DataFrame()

    fs_tmp = 4
    data = []
    for date in tqdm(
        pd.date_range(t_start.date(), today.date(), freq="D").tz_localize(
            "europe/brussels"
        )
    ):
        df_day = df_tmp[date : date + pd.Timedelta(days=1)]
        #         print(date, df_day)

        if not len(df_day):
            # TODO: maybe add empty slot for that day
            continue

        gaps_start = df_day.index.to_series().diff() > timedelta(seconds=1.1 / fs_tmp)
        gaps_start.iloc[[0]] = True

        gaps_end = gaps_start.shift(-1)
        gaps_end.iloc[[-1]] = True

        gaps_start = df_day[gaps_start].index.to_list()
        gaps_end = df_day[gaps_end].index.to_list()
        data += [
            [patient_id, date, gap_start, gap_end]
            for gap_start, gap_end in zip(gaps_start, gaps_end)
        ]

    user_sessions = pd.DataFrame(
        data=data, columns=["patient_id", "date", "start", "end"]
    )
    user_sessions["session_dur"] = user_sessions.end - user_sessions.start

    user_sessions = pd.merge(
        user_sessions,
        user_sessions.groupby("date")["session_dur"].sum().rename("daily_wear_time"),
        left_on="date",
        right_index=True,
        how="outer",
    )
    return user_sessions


def get_stress_interaction_df(
    patient_id: str, user_id: str, window_days=30
) -> pd.DataFrame:
    # query the events for the given range
    today = datetime.now(tz=pytz.timezone("Europe/Brussels")).replace(**daytime_to_0)
    t_start = today - timedelta(days=window_days)

    df_event = pd.json_normalize(
        ContextAwareAPI.get_events_user_interval(user_id, t_start, today)
    )

    # analyze the timeline events
    # 1. Stress events
    df_stress_events = EventDumpParser.parse_stress_events(df_event)

    if not len(df_stress_events):
        # a.warning(
        #     module='interaction rate',
        #     warning_text=f'user {patient_id} - no stress events for the past ' +
        #                  f'{window_days}days'
        # )
        return pd.DataFrame()

    df_stress_events = df_stress_events.set_index("time", drop=True)

    for c in ["prediction", "confirmed", "deprecated"]:
        if c not in df_stress_events.columns:
            df_stress_events[c] = np.NaN

    df_stress_metadata_day = []
    for day in pd.date_range(start=t_start.date(), end=today.date(), freq="D"):
        # rate = (manual events + confirmed predicted events) / total_predicted events
        # with a daily mask

        day_mask = df_stress_events.index.date == day.date()

        if not sum(day_mask):
            df_stress_metadata_day.append(
                [
                    patient_id,
                    day.date(),
                    0,
                    0,
                    0,
                ]
            )
            continue

        df_stress_day = df_stress_events[day_mask]

        nb_user_confirmations = len(
            df_stress_day[
                df_stress_day.prediction
                & df_stress_day.confirmed
                & (df_stress_day.deprecated != True)
            ]
        )

        nb_manual_events = 0
        if "deprecation_contexts" in df_stress_day.columns:
            # deprecated stress events "BY USER" + context -> set stress level = 0
            deprecated_events = len(
                df_stress_day[
                    (df_stress_day.deprecated_by == "user")
                    & (df_stress_day.deprecation_contexts.notna())
                ]
            )
            nb_manual_events += deprecated_events

        nb_user_deleted_events = 0
        if "deprecated_by" in df_stress_day.columns:
            nb_user_deleted_events = int(sum(df_stress_day.deprecated_by == "user"))

        nb_manual_events += len(
            df_stress_day[
                # prediction and deprecated can thus be Nan
                (df_stress_day.prediction != True)
                & (df_stress_day.deprecated != False)
            ]
        )
        nb_timeline_events = len(
            df_stress_day[df_stress_day.prediction & (df_stress_day.deprecated != True)]
        )
        df_stress_metadata_day.append(
            [
                patient_id,
                day.date(),
                nb_user_confirmations,
                nb_manual_events,
                nb_user_deleted_events,
                nb_timeline_events,
            ]
        )
    return pd.DataFrame(
        data=df_stress_metadata_day,
        columns=[
            "patient_id",
            "date",
            "nb_confirmed",
            "nb_manual_inserted",
            "nb_manual_deleted",
            "nb_timeline",
        ],
    )


def get_activity_interaction_df(
    patient_id: str, user_id: str, window_days=30
) -> pd.DataFrame:
    # query the events for the given range
    today = datetime.now(tz=pytz.timezone("Europe/Brussels")).replace(**daytime_to_0)
    t_start = today - timedelta(days=window_days)

    df_event = pd.json_normalize(
        ContextAwareAPI.get_events_user_interval(user_id, t_start, today)
    )

    # analyze the timeline events
    df_activity_events = EventDumpParser.parse_activity_events(df_event)
    if not len(df_activity_events):
        # a.warning(
        #     module='interaction rate',
        #     warning_text=f'user {patient_id} - no activity events for the past ' +
        #                  f'{window_days}days'
        # )
        return pd.DataFrame()

    df_activity_events = df_activity_events.set_index("time", drop=True)
    for c in [
        "endTime",
        "predictedEndTime",
        "feedbackTime",
        "sedentaryConfirmed",
        "deprecated",
    ]:
        if c not in df_activity_events.columns:
            df_activity_events[c] = np.NaN

    df_activity_events["prediction"] = (
        df_activity_events["endTime"] == df_activity_events["predictedEndTime"]
    )

    df_activity_metadata_day = []
    for day in pd.date_range(start=t_start.date(), end=today.date(), freq="D"):
        # rate = (manual events + confirmed predicted events) / total_predicted events
        # with a daily mask
        day_mask = df_activity_events.index.date == day.date()

        if not sum(day_mask):
            df_activity_metadata_day.append(
                [
                    patient_id,
                    day.date(),
                    0,
                    0,
                    0,
                ]
            )
            continue

        df_activity_day = df_activity_events[day_mask]

        nb_user_confirmations = len(
            df_activity_day[
                df_activity_day.prediction
                & (
                    df_activity_day["feedbackTime"].notna()
                    | df_activity_day["sedentaryConfirmed"]
                )
                & (df_activity_day.deprecated != True)
            ]
        )

        nb_user_deleted_events = 0
        if "deprecated_by" in df_activity_day.columns:
            nb_user_deleted_events = int(sum(df_activity_day.deprecated_by == "user"))

        nb_manual_events = 0
        nb_manual_events += len(
            df_activity_day[
                # prediction and deprecated can thus be Nan
                (df_activity_day.prediction != True)
                & (df_activity_day.deprecated != False)
            ]
        )
        nb_timeline_events = len(
            df_activity_day[
                df_activity_day.prediction & (df_activity_day.deprecated != True)
            ]
        )
        df_activity_metadata_day.append(
            [
                patient_id,
                day.date(),
                nb_user_confirmations,
                nb_manual_events,
                nb_user_deleted_events,
                nb_timeline_events,
            ]
        )
    return pd.DataFrame(
        data=df_activity_metadata_day,
        columns=[
            "patient_id",
            "date",
            "nb_confirmed",
            "nb_manual_inserted",
            "nb_manual_deleted",
            "nb_timeline",
        ],
    )


# ---------------------------- Main logic ----------------------------
def controller():
    dt = datetime.now(tz=pytz.timezone("Europe/Brussels"))
    current_time = int(dt.timestamp() * 1000)
    start_of_day = int(dt.replace(**daytime_to_0).timestamp() * 1000)

    for user in (
        seq(ContextAwareAPI.get_active_ids())
        .filter(lambda x: x["patient_id"].startswith("MBRAIN21-"))
        .to_list()
    ):

        user_info = ContextAwareAPI.get_user_info(user["user_id"])
        try:
            print(f"{user['patient_id']}:")

            df_json_events = pd.json_normalize(
                ContextAwareAPI.get_events_user_interval(
                    user_id=user["user_id"], t_start=dt - timedelta(days=120)
                )
            )

            # get the user headache_events
            df_user_headache = EventDumpParser.parse_headache_events(df_json_events)
            if "deprecated" in df_user_headache.columns:
                # only retain the non-deprecated events
                df_user_headache = df_user_headache[df_user_headache.deprecated != True]
            if "intensity" in df_user_headache.columns:
                df_user_headache = df_user_headache[df_user_headache.intensity.notna()]
            else:  # if intensity not in columns -> all nans -> no relevant data
                df_user_headache = pd.DataFrame()

            # get the user questionnaire events
            df_user_questionnaire = EventDumpParser.parse_questionnaire_events(
                df_json_events
            )

            # get the user wear time statistics
            df_user_wear_time = get_user_wear_time_df(
                patient_id=user["patient_id"], window_days=120
            )

            # get the daily records
            df_daily_record = EventDumpParser.parse_daily_record(df_json_events)

            # check the activity and stress interaction rate
            df_activity_interaction = get_activity_interaction_df(
                patient_id=user["patient_id"], user_id=user["user_id"], window_days=120
            )

            df_stress_interaction = get_stress_interaction_df(
                patient_id=user["patient_id"],
                user_id=user["user_id"],
                window_days=120,
            )

            # create a plotly figure with the interaction rate statistics
            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                subplot_titles=[
                    "user wearable sessions ",
                    "daily records",
                    "stress timeline interactions",
                    "activity timeline_interactions",
                ],
            )
            fig.update_yaxes(type="date", tickformat="%H:%M", row=1, col=1)
            fig.update_yaxes(type="date", tickformat="%H:%M", row=2, col=1)
            fig.update_yaxes(title="interaction rate (%)", row=3, col=1)
            fig.update_yaxes(title="interaction rate (%)", row=4, col=1)

            fig.update_layout(
                height=900, title=f"Interaction rate analysis {user}", title_x=0.5
            )

            # 1. add the user wear time
            legend_names = []
            to_same_date = lambda time: datetime.combine(dt.date(), time)
            for _, session in df_user_wear_time.iterrows():
                hours, remainder = divmod(session.daily_wear_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)

                name = "too short " if session.daily_wear_time < timedelta(hours=6) else ""
                name += f"sessions - {user}"

                fig.add_trace(
                    go.Scattergl(
                        x=[session.date] * 2,
                        y=[
                            to_same_date(session.start.time()),
                            to_same_date(session.end.time()),
                        ],
                        line=dict(
                            color=(
                                "rgba(44, 160, 44, 0.3)"  # cooked asparagus green
                                if session.daily_wear_time > timedelta(hours=6)
                                else "rgba(255, 127, 14, 0.4)"  # safety orange
                            ),
                            width=7,
                        ),
                        mode="markers+lines",
                        marker=dict(
                            line_color="black",
                            line_width=1,
                            color="black",
                            symbol="line-ew",
                            size=6,
                            opacity=0.8,
                        ),
                        name=name,
                        legendgroup=user,
                        showlegend=name not in legend_names,
                        visible="legendonly",
                        hovertext=f"total daily wear time = {int(hours)}u{int(minutes)}",
                    ),
                    row=1,
                    col=1,
                )
                legend_names.append(name)

            # 2. Daily records & headache events
            colors = [
                "#a6cee3",
                "#1f78b4",
                "#b2df8a",
                "#33a02c",
                "#fb9a99",
                "#e31a1c",
                "#fdbf6f",
                "#ff7f00",
                "#cab2d6",
                "#6a3d9a",
                "#ffff99",
                "#b15928",
                "#7fc97f",
                "#beaed4",
                "#fdc086",
                "#ffff99",
                "#386cb0",
                "#f0027f",
                "#bf5b17",
                "#666666",
            ]
            if len(df_daily_record):
                df_daily_record = df_daily_record.set_index("date").sort_index()

                # add the food intakes
                try:
                    for i, (col, hovertext) in enumerate(
                        [
                            ("foodIntake.dinnerTime", "dinner"),
                            ("foodIntake.lunchTime", "lunch"),
                            ("foodIntake.breakfastTime", "breakfast"),
                        ]
                    ):
                        if col in df_daily_record.columns:
                            fig.add_trace(
                                go.Scattergl(
                                    x=df_daily_record.index,
                                    name=name,
                                    mode="markers",
                                    y=list(
                                        seq(df_daily_record).map(
                                            lambda x: to_same_date(x.time())
                                            if not pd.isnull(x) and isinstance(x, datetime)
                                            else None
                                        )
                                    ),
                                    line=dict(color=colors[i % len(colors)]),
                                    hovertext=hovertext,
                                    marker_size=8,
                                    legendgroup=user,
                                    visible="legendonly",
                                    showlegend=name not in legend_names,
                                ),
                                row=2,
                                col=1,
                            )
                            legend_names.append(name)
                except Exception as e:
                    print("EXCEPTION", e)
                    print(user)
                    print(df_daily_record.info())

                for col, time in [
                    ("foodIntake.skippedBreakfast", "8:30"),
                    ("foodIntake.skippedLunch", "12:30"),
                    ("foodIntake.skippedDinner", "19:30"),
                ]:
                    if col in df_daily_record.columns:
                        h, m = list(map(int, time.split(":")))
                        df_skipped = df_daily_record[df_daily_record[col] == True]
                        fig.add_trace(
                            go.Scattergl(
                                x=df_skipped.index,
                                y=[to_same_date(dt.replace(hour=h, minute=m).time())]
                                * len(df_skipped),
                                mode="markers",
                                marker_size=8,
                                marker=dict(symbol="x", color="red"),
                                legendgroup=user,
                                hovertext=col.split(".")[-1],
                                name=name,
                                showlegend=name not in legend_names,
                                visible="legendonly",
                            ),
                            row=2,
                            col=1,
                        )

        df_user_quest = df_user_questionnaires[df_user_questionnaires.user == user]
        if len(df_user_quest):
            for q_type, color_str, marker_style in [
                ("morning_nl", "rgba(57, 57, 75, 0.55)", "triangle-up"),
                ("evening_nl", "rgba(57, 57, 75, 0.55)", "triangle-down"),
                ("stress_nl", "rgba(99, 99, 225, 0.3)", "hash"),
                ("stress_misprediction_nl", "rgba(99, 99, 225, 0.3)", "hash"),
                ("nostress_nl", "rgba(99, 99, 225, 0.3)", "hash"),
            ]:
                df_user_q_type = df_user_quest[df_user_quest["payload.name"] == q_type]
                for _, r in df_user_q_type.iterrows():
                    # print(q_type, r['time'].date(),
                    #       [to_same_date(r['payload.start_time'].time()),
                    #        to_same_date(r['payload.end_time'].time())])
                    fig_dict[user].add_trace(
                        go.Scattergl(
                            x=[r["time"].date()],
                            y=[to_same_date(r["payload.end_time"].time())],
                            line=dict(
                                color=color_str,
                            ),
                            mode="markers",
                            marker=dict(
                                line_color="black",
                                line_width=1,
                                color=color_str,
                                symbol=marker_style,
                                size=7,
                                opacity=0.5,
                            ),
                            name=f"questionnaire - {user}",
                            legendgroup=user,
                            showlegend=False,
                            visible="legendonly",
                            hovertext=q_type,
                        ),
                        row=2,
                        col=1,
                    )
                    legend_names.append(f"questionnaires - {user}")

        df_user_headache = df_user_headaches[df_user_headaches.user == user]
        if len(df_user_headache):
            for _, r in df_user_headache.iterrows():
                td = r.endTime - r.time
                hours = int(td.total_seconds() // 3600)
                minutes = int((td.total_seconds() - 3600 * hours) // 60)
                time_str = (
                    f" Periode: {r.time.strftime('%H:%M')} - "
                    f"{r.endTime.strftime('%H:%M')}" + f" - duur: {hours}h{minutes}min"
                )
                intensity_str = "intensiteit: " + str(r.intensity)
                medication_str = (
                    f"Nam medicatie: {str(r.tookMedication)}  -  "
                    + f"medicatie werkte: {str(r.medicationWorked)}<br>"
                )

                if isinstance(r.location, list):
                    loc_str = (
                        "Locatie:<br> * "
                        + "<br> * ".join(
                            [
                                item.get("name_nl", "")
                                for item in r.location
                                if isinstance(item, dict)
                            ]
                        )
                        + "<br>"
                    )
                else:
                    loc_str = "<br> no location <br>"
                if isinstance(r.symptoms, list):
                    symp_str = (
                        "Symptomen:<br> * "
                        + "<br> * ".join(
                            s.get("name_nl")
                            for s in r.symptoms
                            if s.get("isChecked", False)
                        )
                        + "<br>"
                    )
                else:
                    symp_str = "<br> no symptoms <br>"

                if isinstance(r.triggers, list):
                    trigger_str = (
                        "Triggers:<br> * "
                        + "<br> * ".join(
                            s.get("name_nl")
                            for s in r.triggers
                            if s.get("isChecked", False)
                        )
                        + "<br>"
                    )
                else:
                    trigger_str = "<br> no triggers <br>"

                hovertext = "<br>".join(
                    [
                        time_str,
                        intensity_str,
                        medication_str,
                        loc_str,
                        trigger_str,
                        symp_str,
                    ]
                )

                fig_dict[user].add_trace(
                    go.Scatter(
                        x=[r.date] * 2,
                        y=[to_same_date(r.time.time()), to_same_date(r.endTime.time())],
                        line=dict(
                            color="rgba(255, 0, 0, 0.4)",  # red
                            width=7,
                        ),
                        mode="markers+lines",
                        marker=dict(
                            line_color="black",
                            line_width=1,
                            color="black",
                            symbol="line-ew",
                            size=6,
                            opacity=0.8,
                        ),
                        name=f"headaches - {user}",
                        legendgroup=user,
                        showlegend=f"headaches - {user}" not in legend_names,
                        visible="legendonly",
                        hovertext=hovertext,
                    ),
                    row=2,
                    col=1,
                )
                legend_names.append(f"headaches - {user}")

    # 3. Stress interaction rate
    df_stress_interaction["interaction_rate"] = 100 * np.clip(
        (
            df_stress_interaction["nb_confirmed"]
            + df_stress_interaction["nb_manual_inserted"]
        )
        / np.clip(df_stress_interaction["nb_timeline"], a_min=1, a_max=None),
        a_min=0,
        a_max=1,
    )
    df_activity_interaction["interaction_rate"] = 100 * np.clip(
        (
            df_activity_interaction["nb_confirmed"]
            + df_activity_interaction["nb_manual_inserted"]
        )
        / np.clip(df_activity_interaction["nb_timeline"], a_min=1, a_max=None),
        a_min=0,
        a_max=1,
    )

    def check_prediction_interaction(
        df_to_check, interaction_threshold, days_threshold: int, df_name: str
    ):
        week_str = ["MA", "DI", "WOE", "DO", "VR", "ZA", "ZO"]
        df_low = df_to_check[
            (df_to_check.patient_id == user)
            & (df_to_check["interaction_rate"] < interaction_threshold)
            & (df_to_check["nb_timeline"] > 0)
        ].copy()
        # print(df_low.date)
        df_low["date"] = pd.to_datetime(df_low["date"])
        df_low["weekday"] = df_low.date.dt.weekday.map(lambda x: week_str[x])
        df_low["date"] = df_low.date.dt.date

    for i, user in enumerate(
        set(  #
            np.concatenate(
                (
                    df_stress_interaction.patient_id.unique(),
                    df_activity_interaction.patient_id.unique(),
                )
            )
        )
    ):
        name = f"interactions - {user}"

        if user.lower().startswith("mbrain21"):
            print("\t checking stress interaction rate")
            check_prediction_interaction(
                df_stress_interaction,
                interaction_threshold=80,
                days_threshold=5,
                df_name="stress",
            )

            print("\t checking activity interaction rate")
            check_prediction_interaction(
                df_activity_interaction,
                interaction_threshold=80,
                days_threshold=5,
                df_name="activity",
            )

        # Stress interaction visualizations
        df_stress_interaction_user = df_stress_interaction[
            df_stress_interaction.patient_id == user
        ]
        if len(df_stress_interaction_user):
            fig_dict[user].add_trace(
                go.Scattergl(
                    x=df_stress_interaction_user.date,
                    y=df_stress_interaction_user.interaction_rate,
                    mode="markers+lines",
                    name=name,
                    # same color as the activity interactions
                    line=dict(color=colors[i % len(colors)]),
                    hovertext=[
                        "<br>".join(
                            [
                                f"total_events {session.nb_timeline + session.nb_manual_inserted}",
                                f"nbr_manual_inserted: {session.nb_manual_inserted}",
                                f"nbr_manual_deleted: {session.nb_manual_deleted}",
                                f"nbr_confirmed: {session.nb_confirmed}",
                            ]
                        )
                        for _, session in df_stress_interaction_user.iterrows()
                    ],
                    legendgroup=user,
                    showlegend=name not in legend_names,
                    visible="legendonly",
                ),
                row=3,
                col=1,
            )
            legend_names.append(name)

        # Activity interaction visualizations
        df_activity_interaction_user = df_activity_interaction[
            df_activity_interaction.patient_id == user
        ]
        if len(df_activity_interaction_user):
            fig_dict[user].add_trace(
                go.Scattergl(
                    x=df_activity_interaction_user.date,
                    y=df_activity_interaction_user.interaction_rate,
                    mode="markers+lines",
                    name=name,
                    # same color as the stress interactions
                    line=dict(color=colors[i % len(colors)]),
                    hovertext=[
                        "<br>".join(
                            [
                                f"total_events {session.nb_timeline + session.nb_manual_inserted}",
                                f"nbr_manual_inserted: {session.nb_manual_inserted}",
                                f"nbr_manual_deleted: {session.nb_manual_deleted}",
                                f"nbr_confirmed: {session.nb_confirmed}",
                            ]
                        )
                        for _, session in df_activity_interaction_user.iterrows()
                    ],
                    legendgroup=user,
                    visible="legendonly",
                    showlegend=name not in legend_names,
                ),
                row=4,
                col=1,
            )
            legend_names.append(name)

        fig_dict[user].add_hline(y=80, line_dash="dash", row=3, col=1)
        fig_dict[user].add_hline(y=80, line_dash="dash", row=4, col=1)

    # save the visualization into the HTML and upload it to idlab cloud
    for user, fig in fig_dict.items():
        save_path = f'./interaction_rate_{user}_{dt.strftime("%Y-%m-%d__%H:%M")}.html'
        figs_to_html([fig], save_path)


if __name__ == "__main__":
    controller()
