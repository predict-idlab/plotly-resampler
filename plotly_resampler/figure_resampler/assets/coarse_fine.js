function getGraphDiv(gdID) {
    let graphDiv = document?.querySelectorAll('div[id*="' + gdID + '"][class*="dash-graph"]');
    graphDiv = graphDiv?.[0]?.getElementsByClassName("js-plotly-plot")?.[0];
    if (!_.isElement(graphDiv)) {
        throw new Error(`Invalid gdID '${gdID}'`);
    }
    return graphDiv;
}

/**
 *
 * @param {object} data The data of the graphDiv
 * @returns {Array} An array containing all the unique axis keys of the graphDiv data
 *                  [{x: x[ID], y: y[ID]}, {x: x[ID], y: y[ID]}]
 */
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

/**
 * Get the corresponding axis name of the anchors
 *
 * @param {object} layout the layout of the graphDiv
 * @returns {object} An object containing the anchor and its orthogonal axis name e.g.
 *                  {x[ID]: yaxis[ID], y[ID]: xaxis[ID]}
 */
const getLayoutAxisAnchors = (layout) => {
    var layout_axis_anchors = Object.assign(
        {},
        ..._.chain(layout)
            .map((value, key) => {
                if (key.includes("axis")) return { [value.anchor]: key };
            })
            .without(undefined)
            .value()
    );
    // Edge case for non "make_subplot" figures; i.e. figures constructed with
    // go.Figure
    if (_.size(layout_axis_anchors) == 1 && _.has(layout_axis_anchors, undefined)) {
        return { x: "yaxis", y: "xaxis" };
    }
    return layout_axis_anchors;
};

/**
 * Compare the equality of two arrays with a certain decimal point presiction
 * @param {*} objValueArr An array with numeric values
 * @param {*} othValueArr An array with numeray values
 * @returns {boolean} true when all values are equal (to 5 decimal points)
 */
function rangeCustomizer(objValueArr, othValueArr) {
    return _.every(
        _.zipWith(objValueArr, othValueArr, (objValue, othValue) => {
            if (_.isNumber(objValue) && _.isNumber(othValue)) {
                objValue = _.round(objValue, 5);
                othValue = _.round(othValue, 5);
                return objValue === othValue;
            } else {
                alert(`not a number  ${objValue} type:${typeof objValue} | ${othValue} type:${typeof othValue}`);
            }
        })
    );
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        coarse_to_main: function (selectedData, mainFigID, coarseFigID) {
            // Base case
            if (!selectedData.range) {
                return mainFigID;
            }

            main_graphDiv = getGraphDiv(mainFigID);
            coarse_graphDiv = getGraphDiv(coarseFigID);

            const coarse_xy_axiskeys = getXYAxisKeys(coarse_graphDiv.data);
            const main_xy_axiskeys = getXYAxisKeys(main_graphDiv.data);
            const layout_axis_anchors = getLayoutAxisAnchors(main_graphDiv.layout);

            // Use the maingraphDiv its layout to obtain a list of a list of all shared (x)axis names
            // in practice, these are the xaxis names that are linked to each other (i.e. the inner list is the
            // xaxis names of the subplot columns)
            // e.g.: [ [xaxis1, xaxis2],  [xaxis3, xaxis4] ]
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
                )
                // add the axis string to the match array and return the match array
                .map((m_obj) => {
                    const anchorT = getAnchorT(main_xy_axiskeys, m_obj.anchor);
                    let axis_str = layout_axis_anchors[anchorT];
                    m_obj.match.push(axis_str);
                    return m_obj.match;
                })
                .value();
            // console.log("shared axes list", shared_axes_list);

            const relayout = {};

            // Quick inline function to set the relayout range values
            const setRelayoutRangeValues = (axisStr, values) => {
                for (let rangeIdx = 0; rangeIdx < 2; rangeIdx++) {
                    relayout[axisStr + `.range[${rangeIdx}]`] = values[rangeIdx];
                }
            };

            // iterate over the selected data range
            // console.log("selected data range", selectedData.range);
            for (const anchor_key in selectedData.range) {
                const selected_range = selectedData.range[anchor_key];
                // Obtain the anchor key of the orthogonal axis (x or y), based on the coarse graphdiv anchor pairs
                const anchorT = getAnchorT(coarse_xy_axiskeys, anchor_key);
                const axisStr = layout_axis_anchors[anchorT];
                const mainLayoutRange = main_graphDiv.layout[axisStr].range;
                const coarseFigRange = coarse_graphDiv.layout[axisStr].range;

                if (!_.isEqual(selected_range, mainLayoutRange)) {
                    const shared_axis_match = _.chain(shared_axes_list)
                        .filter((arr) => arr.includes(axisStr))
                        .value()[0];
                    if (axisStr.includes("yaxis") && _.isEqualWith(selected_range, coarseFigRange, rangeCustomizer)) {
                        continue;
                    }

                    if (shared_axis_match) {
                        shared_axis_match.forEach((axisMStr) => {
                            setRelayoutRangeValues(axisMStr, selected_range);
                        });
                    } else {
                        setRelayoutRangeValues(axisStr, selected_range);
                    }
                }
            }

            Object.keys(relayout).length > 0 ? Plotly.relayout(main_graphDiv, relayout) : null;
            return mainFigID;
        },

        main_to_coarse: function (mainRelayout, coarseFigID, mainFigID) {
            const coarse_graphDiv = getGraphDiv(coarseFigID);
            const main_graphDiv = getGraphDiv(mainFigID);

            const coarse_xy_axiskeys = getXYAxisKeys(coarse_graphDiv.data);
            const layout_axis_anchors = getLayoutAxisAnchors(coarse_graphDiv.layout);

            const currentSelections = coarse_graphDiv.layout.selections;
            const update = { selections: currentSelections || [] };

            const getUpdateObj = (xy_pair, x_range, y_range) => {
                return {
                    type: "rect",
                    xref: xy_pair.x,
                    yref: xy_pair.y,
                    line: { width: 1, color: "#352F44", dash: "solid" },
                    x0: x_range[0],
                    x1: x_range[1],
                    y0: y_range[0],
                    y1: y_range[1],
                };
            };

            // Base case; no selections yet on the coarse graph
            if (!currentSelections) {
                // if current selections is None
                coarse_xy_axiskeys.forEach((xy_pair) => {
                    // console.log("xy pair", xy_pair);
                    const x_axis_key = _.has(layout_axis_anchors, xy_pair.y) ? layout_axis_anchors[xy_pair.y] : "xaxis";
                    const y_axis_key = _.has(layout_axis_anchors, xy_pair.x) ? layout_axis_anchors[xy_pair.x] : "yaxis";
                    // console.log('xaxis key', x_axis_key, main_graphDiv.layout[x_axis_key]);
                    const x_range = main_graphDiv.layout[x_axis_key].range;
                    const y_range = main_graphDiv.layout[y_axis_key].range;

                    update["selections"].push(getUpdateObj(xy_pair, x_range, y_range));
                });
                Plotly.relayout(coarse_graphDiv, update);
                return coarseFigID;
            }

            // Alter the selections based on the relayout
            let performed_update = false;

            for (let i = 0; i < coarse_xy_axiskeys.length; i++) {
                const xy_pair = coarse_xy_axiskeys[i];
                // If else handles the edge case of a figure without subplots
                const x_axis_key = _.has(layout_axis_anchors, xy_pair.y) ? layout_axis_anchors[xy_pair.y] : "xaxis";
                const y_axis_key = _.has(layout_axis_anchors, xy_pair.x) ? layout_axis_anchors[xy_pair.x] : "yaxis";
                // console.log('xaxis key', x_axis_key, main_graphDiv.layout[x_axis_key]);

                let x_range = main_graphDiv.layout[x_axis_key].range;
                let y_range = main_graphDiv.layout[y_axis_key].range;
                // If the y-axis autorange is true, we alter the y-range to the coarse graphdiv its y-range
                // console.log('mainrelayout', mainRelayout);
                if (main_graphDiv.layout[y_axis_key]["autorange"] === true) {
                    y_range = coarse_graphDiv.layout[y_axis_key].range;
                }
                if (
                    mainRelayout[x_axis_key + ".autorange"] === true &&
                    mainRelayout[y_axis_key + ".autorange"] === true
                ) {
                    performed_update = true;
                    if (
                        // NOTE: for some reason, showspikes info is only available for the xaxis & yaxis keys
                        _.has(mainRelayout, "xaxis.showspikes") &&
                        _.has(mainRelayout, "yaxis.showspikes")
                    ) {
                        // reset axis -> we use the coarse graphDiv layout
                        x_range = coarse_graphDiv.layout[x_axis_key].range;
                    }
                } else if (mainRelayout[x_axis_key + ".range[0]"] || mainRelayout[y_axis_key + ".range[0]"]) {
                    // a specific range is set
                    performed_update = true;
                }

                update["selections"][i] = getUpdateObj(xy_pair, x_range, y_range);
            }
            performed_update ? Plotly.relayout(coarse_graphDiv, update) : null;
            return coarseFigID;
        },
    },
});
