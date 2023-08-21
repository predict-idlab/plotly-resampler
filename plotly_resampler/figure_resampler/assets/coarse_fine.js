const getXYAxisKeys = (data) => {
    return _.chain(data)
        .map((obj) => ({ x: obj.xaxis || "x", y: obj.yaxis || "y" }))
        .uniqWith(_.isEqual)
        .value();
};

const getAnchorT = (keys, anchor) => {
    const obj_index = anchor.slice(0, 1);
    const anchorT = _.chain(keys)
        .filter((obj) => obj[obj_index] == anchor)
        .value()[0][{ x: "y", y: "x" }[obj_index]];

    return anchorT;
};

const getLayoutAxisAnchors = (layout) => {
    return Object.assign(
        {},
        ..._.chain(layout)
            .map((value, key) => {
                if (key.includes("axis")) return { [value.anchor]: key };
            })
            .without(undefined)
            .value()
    );
};

function getGraphDiv(gdID) {
    let graphDiv = document?.querySelectorAll('div[id*="' + gdID + '"][class*="dash-graph"]');
    graphDiv = graphDiv?.[0]?.getElementsByClassName("js-plotly-plot")?.[0];
    if (!_.isElement(graphDiv)) {
        throw new Error(`Invalid gdID '${gdID}'`);
    }
    return graphDiv;
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        coarse_to_main: function (selectedData, mainFigID, coarseFigID) {
            /* ------------ CALLBACK ----------- */
            main_graphDiv = getGraphDiv(mainFigID);
            coarse_graphDiv = getGraphDiv(coarseFigID);

            // An object containing each layout axis its x and y anchor {x<ID>: xaxis<ID>, y<ID>: yaxis<ID>}
            const coarse_xy_axiskeys = getXYAxisKeys(coarse_graphDiv.data);
            const main_xy_axiskeys = getXYAxisKeys(main_graphDiv.data);
            const layout_axis_anchors = getLayoutAxisAnchors(main_graphDiv.layout);

            let shared_axes_list = _.chain(main_graphDiv.layout)
                .map((value, key) => {
                    if (value.matches) return { anchor: value.matches, match: [key] };
                })
                .without(undefined)
                // groupby same anchor and concat the match arrays
                .groupBy("anchor")
                .map(
                    _.spread((...values) => {
                        return _.mergeWith(...values, (objValue, srcValue) => {
                            if (_.isArray(objValue)) return objValue.concat(srcValue);
                        });
                    })
                );
            // add the axis string to the match array and return the match array
            // console.log("shared axes list", shared_axes_list);
            shared_axes_list = shared_axes_list
                .map((m_obj) => {
                    const anchorT = getAnchorT(main_xy_axiskeys, m_obj.anchor);
                    let axis_str = layout_axis_anchors[anchorT];
                    m_obj.match.push(axis_str);
                    return m_obj.match;
                })
                .value();
            // console.log("shared axes list", shared_axes_list);

            // De x & y anchors van de coarse graphdiv gebruken?
            // console.log("selected data", selectedData);
            // console.log("layout axis anchors", layout_axis_anchors);
            // console.log("coarse xy axis keys", layout_axis_anchors);
            if (!selectedData.range) {
                return mainFigID;
            }

            const relayout = {};
            const setRelayoutRangeValues = (index, values) => {
                for (let i = 0; i < 2; i++) {
                    relayout[index + `.range[${i}]`] = values[i];
                }
            };
            // iterate over the selected data range
            for (const anchor_key in selectedData.range) {
                const selected_range = selectedData.range[anchor_key];

                // NOTE: obtain the anchor key of the other axis (x or y), based on the
                // coarse graphdiv anchor pairs
                const anchorT = getAnchorT(coarse_xy_axiskeys, anchor_key);
                const axis_str = layout_axis_anchors[anchorT];
                const main_layout_range = main_graphDiv.layout[axis_str].range;

                // console.log("anchor key", anchor_key, "\tanchor T", anchorT, "\taxis str", axis_str);
                // console.log("axis_range", selected_range);
                // console.log("main layout range", main_layout_range);
                // console.log(_.isEqual(selected_range, main_layout_range));
                if (!_.isEqual(selected_range, main_layout_range)) {
                    const shared_axis_match = _.chain(shared_axes_list)
                        .filter((arr) => arr.includes(axis_str))
                        .value()[0];
                    if (shared_axis_match) {
                        shared_axis_match.forEach((axis_m_str) => {
                            setRelayoutRangeValues(axis_m_str, selected_range);
                        });
                    } else {
                        setRelayoutRangeValues(axis_str, selected_range);
                    }
                }
            }

            if (Object.keys(relayout).length > 0) {
                Plotly.relayout(main_graphDiv, relayout);
            }
            return mainFigID;
        },
        main_to_coarse: function (mainRelayout, coarseFigID, mainFigID) {
            /* ------------ CALLBACK ----------- */
            const coarse_graphDiv = getGraphDiv(coarseFigID);
            const main_graphDiv = getGraphDiv(mainFigID);
            // console.log("relayout", mainRelayout);

            // An array containing the unique [{x: 'x<ID>', y: 'y<ID>'}] axis keys of the coarse graphDiv trace data
            const coarse_xy_axiskeys = getXYAxisKeys(coarse_graphDiv.data);
            // console.log("coarse xy axis keys", coarse_xy_axiskeys);

            // An object containing each layout axis its x and y anchor {x<ID>: xaxis<ID>, y<ID>: yaxis<ID>}
            const layout_axis_anchors = getLayoutAxisAnchors(coarse_graphDiv.layout);
            // console.log("layout axis anchors", layout_axis_anchors);

            const currentSelections = coarse_graphDiv.layout.selections;
            const update = { selections: currentSelections || [] };

            const getUpdateObj = (xy_pair, x_range, y_range) => {
                return {
                    type: "rect",
                    xref: xy_pair.x,
                    yref: xy_pair.y,
                    line: { width: 1, dash: "dot" },
                    x0: x_range[0],
                    x1: x_range[1],
                    y0: y_range[0],
                    y1: y_range[1],
                };
            };

            // Base case; no selections yet on the coarse graph
            if (!currentSelections) {
                // if current selections is None
                console.log("no selections found");
                coarse_xy_axiskeys.forEach((xy_pair) => {
                    const x_axis_key = layout_axis_anchors[xy_pair.y];
                    const y_axis_key = layout_axis_anchors[xy_pair.x];
                    const x_range = main_graphDiv.layout[x_axis_key].range;
                    const y_range = main_graphDiv.layout[y_axis_key].range;

                    // console.log(x_range, y_range);
                    update["selections"].push(getUpdateObj(xy_pair, x_range, y_range));
                });
                Plotly.relayout(coarse_graphDiv, update);
                return coarseFigID;
            }

            // Alter the selections based on the relayout
            let performed_update = false;

            for (let i = 0; i < coarse_xy_axiskeys.length; i++) {
                const xy_pair = coarse_xy_axiskeys[i];
                const x_axis_key = layout_axis_anchors[xy_pair.y];
                const y_axis_key = layout_axis_anchors[xy_pair.x];
                // console.log(x_axis_key, y_axis_key);

                let x_range = main_graphDiv.layout[x_axis_key].range;
                let y_range = main_graphDiv.layout[y_axis_key].range;
                if (
                    mainRelayout[x_axis_key + ".autorange"] === true &&
                    mainRelayout[y_axis_key + ".autorange"] === true
                ) {
                    performed_update = true;
                    if (
                        mainRelayout[x_axis_key + ".showspikes"] === false &&
                        mainRelayout[y_axis_key + ".showspikes"] === false
                    ) {
                        // reset axis -> we use the coarse graphDiv layout
                        x_range = coarse_graphDiv.layout[x_axis_key].range;
                        y_range = coarse_graphDiv.layout[y_axis_key].range;
                    }
                } else if (mainRelayout[x_axis_key + ".range[0]"] || mainRelayout[y_axis_key + ".range[0]"]) {
                    performed_update = true;
                }
                update["selections"][i] = getUpdateObj(xy_pair, x_range, y_range);
            }

            if (performed_update) {
                Plotly.relayout(coarse_graphDiv, update);
            }
            return coarseFigID;
        },
    },
});
