window.dash_clientside = Object.assign({}, window.dash_clientside, {
	clientside: {
		// TODO -> fix doubble callback -> maybe check whether the range of the selected that is the same 
		// range of the figure?
		coarse_to_main: function (selectedData, mainFigID, coarseFigID, linkedIndices) {
			//define helper function (imports not allowed?)
			function getGraphDiv(gdID) {
				// see this link for more information https://stackoverflow.com/a/34002028 
				let graphDiv = document?.querySelectorAll('div[id*="' + gdID + '"][class*="dash-graph"]');
				if (graphDiv.length > 1) {
					throw new SyntaxError("UpdateStore: multiple graphs with ID=" + gdID + " found; n=" + graphDiv.length + " (either multiple graphs with same ID's or current ID is a str-subset of other graph IDs)");
				} else if (graphDiv.length < 1) {
					throw new SyntaxError("UpdateStore: no graphs with ID=" + gdID + " found");
				}
				graphDiv = graphDiv?.[0]?.getElementsByClassName('js-plotly-plot')?.[0];
				const isDOMElement = el => el instanceof HTMLElement
				if (!isDOMElement) {
					throw new Error(`Invalid gdID '${gdID}'`);
				}
				return graphDiv;
			}
			//can be changed for something more sophisticated if needed, 
			//it does the job of determining if two arrays have the same values for now
			const compareArrays = (a, b) => {
				return JSON.stringify(a) === JSON.stringify(b);
			};
			const roundTo10Decimals = (a) => {
				return +a.toFixed(10)
			};

			/**
			 * 
			 * @param {object} layout : layout object from which the x and y ranges should be extracted
			 * @param {Array} indexList : list of subplot indices of which the x and y range should be extracted. undefined = all subplots
			 * @returns an array containing an array of xranges and an array of yranges of the figure the layout comes from
			 */
			const getFigureInfo = (layout, indexList = undefined) => {
				const yrange = [];
				const xrange = [];

				Object.keys(layout).forEach(key => {
					if (key.startsWith("yaxis")) {
						const axisNumber = key.slice(5, 6) ? key.slice(5, 6) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							yrange.push({ [axisNumber]: axisRange });
						}
					} else if (key.startsWith("xaxis")) {
						const axisNumber = key.slice(5, 9) ? key.slice(5, 9) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							xrange.push({ [axisNumber]: axisRange });
						}
					}
				});
				return [xrange, yrange];
			};

			const getGridLayout = (layout) => {
				const xaxes = new Set();
				const yaxes = new Set();

				Object.keys(layout).forEach(key => {
					if (key.startsWith("yaxis")) {
						yaxes.add(JSON.stringify(layout[key].domain));
					} else if (key.startsWith("xaxis")) {
						xaxes.add(JSON.stringify(layout[key].domain));
					}
				});
				return [yaxes.size, xaxes.size]
			}

			// console.warn("before coarse -> main");

			//obtain the graph components from the DOM
			main_graphDiv = getGraphDiv(mainFigID);
			coarse_graphDiv = getGraphDiv(coarseFigID);
			let updateCondition = false;
			let updates = {};
			if (selectedData) {
				if (selectedData.range) {
					console.warn("starting: coarse -> main");

					let triggerCols = [...new Set(Object.keys(selectedData.range).map(item => +(item.substring(1) || 1) - 1))];

					const [mainRows, mainCols] = getGridLayout(main_graphDiv.layout);
					// console.log(mainCols);
					// console.log(mainRows);

					//to be passed as argument of this function?
					// let linkedIndices = [1, 1, 2];
					//perform extra checks on passed linkedIndices (could be moved over to the back-end + deleted from here)
					if (mainCols > 0 && linkedIndices.length > mainCols) {
						linkedIndices = linkedIndices.slice(0, mainCols);
					} else if (mainCols === 0) {
						linkedIndices = [linkedIndices[0]];
					}

					linkedIndices = linkedIndices.map((item, i) => {
						if (mainRows === 0) {
							item = 0;
						} else if (item >= mainRows) {
							// set the row to the max possible index within the grid
							item = mainRows - 1;
						}
						return item;
					})

					const updateData = [];
					linkedIndices.forEach((item, i) => {
						updateData.push({
							"columnIndex": +i,
							"linkIndex": +item,
							"mainSubplotIndex": +linkedIndices.length * item + i
						})
					})


					let filteredTriggerData = updateData.filter(obj => triggerCols.includes(obj.columnIndex));

					const filteredTriggers = filteredTriggerData.map(obj => obj.mainSubplotIndex);

					let [xrange, yrange] = getFigureInfo(main_graphDiv.layout, filteredTriggers);
					// let [cxrange, cyrange] = getFigureInfo(coarse_graphDiv.layout);

					filteredTriggerData = filteredTriggerData.map(obj => {
						const subplotIndex = obj.mainSubplotIndex;
						// const columnIndex = obj.columnIndex;
						const yrangeval = yrange.find(o => o.hasOwnProperty(subplotIndex));
						const xrangeval = xrange.find(o => o.hasOwnProperty(subplotIndex));
						// const cyrangeval = cyrange.find(o => o.hasOwnProperty(columnIndex));
						// const cxrangeval = cxrange.find(o => o.hasOwnProperty(columnIndex));

						if (xrangeval) {
							obj["mxrange"] = xrangeval[subplotIndex];
							obj["myrange"] = yrangeval[subplotIndex].map(roundTo10Decimals);
							// obj["cxrange"] = cxrangeval[columnIndex];
							// obj["cyrange"] = cyrangeval[columnIndex].map(roundTo10Decimals);
							obj["sxrange"] = [];
							obj["syrange"] = [];
						}
						return obj
					});
					console.log(filteredTriggerData);


					let currentSelections = coarse_graphDiv.layout.selections;
					console.log(currentSelections);
					if (currentSelections) {

						filteredTriggerData = filteredTriggerData.map(obj => {
							const subplotIndex = obj.columnIndex + 1;
							const filteredSelection = currentSelections.filter(selection => {
								//Plotly accepts both x and x1 as indices, 
								//so if we are looking for selections the first axis, check both options
								if (obj.columnIndex == 0) {
									return selection.xref === 'x' || selection.xref === `x${subplotIndex}`;
								} else {
									return selection.xref === `x${subplotIndex}`
								}

							});
							//filteredSelection and currentSelection sxrange is not the same?????????????
							console.log(filteredSelection);

							if (filteredSelection.length > 0) {
								obj["sxrange"] = [filteredSelection[0].x0, filteredSelection[0].x1].sort((a, b) => Date.parse(a) - Date.parse(b));
								obj["syrange"] = [filteredSelection[0].y0, filteredSelection[0].y1].sort(function (a, b) { return a - b }).map(roundTo10Decimals);
							}
							return obj
						});
					}

					/**
					 * 1. check which graph the selection change comes from
					 */


					/* 2 conditions for an update of the main graph: 
						
						1. the selection range is not the same as the main graph's range (before the update).

						2. the selection range is not exactly the same as the coarse graph's range 
						(which should be the same as the original main graph's range). 
						A selection range like this will be linked to the reset of the axes, meaning it comes from the 
						"main -> coarse" (update_coarse_selectbox) callback.

						DIFFERENT RELAYOUT MESSAGES: if the selection xrange is the same as the main graph xrange, 
						but this is not the case in the y direction, relayout should only contain the y range change.
						This is done to prevent an unnecessary aggregation in the Plotly resampler back-end
					*/


					filteredTriggerData.forEach((obj, i) => {
						// console.log(!compareArrays(obj.sxrange, obj.mxrange));
						// console.log(!compareArrays(obj.syrange, obj.myrange));
						// console.log(!compareArrays(obj.sxrange, obj.cxrange));
						// console.log(!compareArrays(obj.syrange, obj.cyrange));
						// console.log(obj.mainSubplotIndex+1);
						const subplotIndex = +obj.mainSubplotIndex === 0 ? "" : String(obj.mainSubplotIndex + 1);
						// console.log(subplotIndex);
						// console.log(`xaxis${subplotIndex}.range[0]`)

						let xaxisKey0 = `xaxis${subplotIndex}.range[0]`;
						let xaxisKey1 = `xaxis${subplotIndex}.range[1]`;
						let yaxisKey0 = `yaxis${subplotIndex}.range[0]`;
						let yaxisKey1 = `yaxis${subplotIndex}.range[1]`;

						if ((!compareArrays(obj.sxrange, obj.mxrange) && !compareArrays(obj.syrange, obj.myrange))) {
							updateCondition = true;
							updates = {
								...updates,
								[xaxisKey0]: obj.sxrange[0],
								[xaxisKey1]: obj.sxrange[1],
								[yaxisKey0]: obj.syrange[0],
								[yaxisKey1]: obj.syrange[1],
							};
						} else if (!compareArrays(obj.sxrange, obj.mxrange)) {
							updateCondition = true
							updates = {
								...updates,
								[xaxisKey0]: obj.sxrange[0],
								[xaxisKey1]: obj.sxrange[1],
							};
						} else if (!compareArrays(obj.syrange, obj.myrange)) {
							updateCondition = true
							updates = {
								...updates,
								[yaxisKey0]: obj.syrange[0],
								[yaxisKey1]: obj.syrange[1],
							};
						}

					});
					console.log(updates);

				}
				// else {
				// 	console.log('selectedData.range null? -> ' + selectedData + ' reset main graph');
				// 	updateCondition = true;
				// 	updates = { ...updates,
				// 		'xaxis.autorange': true, 'xaxis.showspikes': false,
				// 		'yaxis.autorange': true, 'yaxis.showspikes': false,
				// 	};
				// }
			}
			if (updateCondition) {
				// Update the layout without triggering another relayout event
				console.warn('coarse -> main');
				// document.dispatchEvent(); doesnt work
				Plotly.relayout(
					main_graphDiv, updates
				);
			}

			return mainFigID;
		},
		main_to_coarse: function (mainFigRelayout, coarseFigID, mainFigID, linkedIndices) {
			//define helper function (imports not allowed?)
			function getGraphDiv(gdID) {
				// see this link for more information https://stackoverflow.com/a/34002028 
				let graphDiv = document?.querySelectorAll('div[id*="' + gdID + '"][class*="dash-graph"]');
				if (graphDiv.length > 1) {
					throw new SyntaxError("UpdateStore: multiple graphs with ID=" + gdID + " found; n=" + graphDiv.length + " (either multiple graphs with same ID's or current ID is a str-subset of other graph IDs)");
				} else if (graphDiv.length < 1) {
					throw new SyntaxError("UpdateStore: no graphs with ID=" + gdID + " found");
				}
				graphDiv = graphDiv?.[0]?.getElementsByClassName('js-plotly-plot')?.[0];
				const isDOMElement = el => el instanceof HTMLElement
				if (!isDOMElement) {
					throw new Error(`Invalid gdID '${gdID}'`);
				}
				return graphDiv;
			}
			//can be changed for something more sophisticated if needed, 
			//it does the job of determining if two arrays have the same values for now
			const compareArrays = (a, b) => {
				return JSON.stringify(a) === JSON.stringify(b);
			};

			const roundTo10Decimals = (a) => {
				return +a.toFixed(10)
			};

			/**
			 * 
			 * @param {object} layout : layout object from which the x and y ranges should be extracted
			 * @param {Array} indexList : list of subplot indices of which the x and y range should be extracted. undefined = all subplots
			 * @returns an array containing an array of xranges and an array of yranges of the figure the layout comes from
			 */
			const getFigureInfo = (layout, indexList = undefined) => {
				const yrange = [];
				const xrange = [];

				Object.keys(layout).forEach(key => {
					if (key.startsWith("yaxis")) {
						const axisNumber = key.slice(5, 6) ? key.slice(5, 6) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							yrange.push({ [axisNumber]: axisRange });
						}
					} else if (key.startsWith("xaxis")) {
						const axisNumber = key.slice(5, 9) ? key.slice(5, 9) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							xrange.push({ [axisNumber]: axisRange });
						}
					}
				});
				return [xrange, yrange];
			};

			const getGridLayout = (layout) => {
				const xaxes = new Set();
				const yaxes = new Set();

				Object.keys(layout).forEach(key => {
					if (key.startsWith("yaxis")) {
						yaxes.add(JSON.stringify(layout[key].domain));
					} else if (key.startsWith("xaxis")) {
						xaxes.add(JSON.stringify(layout[key].domain));
					}
				});
				return [yaxes.size, xaxes.size]
			}

			//function to extract the subplot index from which the relayout comes from
			const getTriggerSubplot = (relayout) => {
				let triggerPlot = new Set();
				const patternRange = /(y|x)?axis|\.range\[\d\]/g;
				const patternAutorange = /(y|x)?axis|\.autorange/g;
				Object.keys(relayout).forEach(key => {
					if (key.startsWith("yaxis") || key.startsWith("xaxis")) {
						let axisNumber;
						if (key.includes(".range")) {
							axisNumber = key.replace(patternRange, '');
							axisNumber = axisNumber ? +axisNumber - 1 : 0;
							triggerPlot.add(axisNumber);
						} else if (key.includes(".autorange")) {
							axisNumber = key.replace(patternAutorange, '');
							axisNumber = axisNumber ? +axisNumber - 1 : 0;
							triggerPlot.add(axisNumber);
						}
					}
				});
				return Array.from(triggerPlot);
			}
			// external_scripts=[{'src':'../utils/callback_utils','type':'module'}]


			console.warn('starting: main -> coarse');
			console.log(mainFigRelayout);

			let coarse_graphDiv = getGraphDiv(coarseFigID);
			let main_graphDiv = getGraphDiv(mainFigID);

			let trigger = getTriggerSubplot(mainFigRelayout);
			// console.log(trigger);
			const [mainRows, mainCols] = getGridLayout(main_graphDiv.layout);

			// linkedIndices is an array showing which subplots of the main graph are linked with a coarse view
			// obtained from a Store in the client (1st idea)
			// structure: index = column, value = row (item at index 1 with value 2 is in the 2nd column, 3rd row)
			// ensures there's only 1 linked subplot per column!
			// const linkedIndices = new Array(coarseCols).fill(0);

			//perform extra checks on passed linkedIndices (could be moved over to the back-end + deleted from here)
			// let linkedIndices = [1, 1, 2];
			if (mainCols > 0 && linkedIndices.length > mainCols) {
				linkedIndices = linkedIndices.slice(0, mainCols);
			} else if (mainCols === 0) {
				linkedIndices = [linkedIndices[0]];
			}

			linkedIndices = linkedIndices.map((item, i) => {
				if (mainRows === 0) {
					item = 0;
				} else if (item >= mainRows) {
					// set the row to the max possible index within the grid
					item = mainRows - 1;
				}
				return item;
			})
			// console.log(linkedIndices);

			//list of objects compiling all data needed to create an update for relayout
			const updateData = [];
			linkedIndices.forEach((item, i) => {
				updateData.push({
					"columnIndex": +i,
					"linkIndex": +item,
					"mainSubplotIndex": +linkedIndices.length * item + i
				})
			})
			// console.log(updateData);

			//filter the trigger list to only the ones that have a coarse view linked to them
			let filteredTriggerData = updateData.filter(obj => trigger.includes(obj.mainSubplotIndex));
			// console.log(filteredTriggerData);

			const filteredTriggers = filteredTriggerData.map(obj => obj.mainSubplotIndex);


			//obtain the x & yrange of the triggered and linked subplots
			let [xrange, yrange] = getFigureInfo(main_graphDiv.layout, filteredTriggers);
			let [cxrange, cyrange] = getFigureInfo(coarse_graphDiv.layout);

			filteredTriggerData = filteredTriggerData.map(obj => {
				const subplotIndex = obj.mainSubplotIndex;
				const columnIndex = obj.columnIndex;
				const yrangeval = yrange.find(o => o.hasOwnProperty(subplotIndex));
				const xrangeval = xrange.find(o => o.hasOwnProperty(subplotIndex));
				const cyrangeval = cyrange.find(o => o.hasOwnProperty(columnIndex));
				const cxrangeval = cxrange.find(o => o.hasOwnProperty(columnIndex));


				if (xrangeval) {
					obj["mxrange"] = xrangeval[subplotIndex].sort((a, b) => Date.parse(a) - Date.parse(b));
					obj["myrange"] = yrangeval[subplotIndex].map(roundTo10Decimals).sort(function (a, b) { return a - b });
					obj["cxrange"] = cxrangeval[columnIndex].sort((a, b) => Date.parse(a) - Date.parse(b));
					obj["cyrange"] = cyrangeval[columnIndex].map(roundTo10Decimals).sort(function (a, b) { return a - b });
					obj["sxrange"] = [];
					obj["syrange"] = [];
				}
				return obj
			});

			// obtain the selections that should be changed (using the subplot index mapped in coarseTriggerMap)
			// create syrange
			// used to check if it really is necessary to change the selectionbox!
			let currentSelections = coarse_graphDiv.layout.selections;
			console.log(currentSelections);
			if (currentSelections) {

				filteredTriggerData = filteredTriggerData.map(obj => {
					const subplotIndex = obj.columnIndex;
					console.log(subplotIndex);
					const filteredSelection = currentSelections.filter(selection => {
						if (subplotIndex === 0) {
							return selection.xref === `x${subplotIndex + 1}` || selection.xref === "x"
						} else {
							return selection.xref === `x${subplotIndex + 1}`
						}

					});

					//filteredSelection and currentSelection sxrange is not the same (approx errors?)?? so confused
					console.log(filteredSelection);

					if (filteredSelection.length > 0) {
						obj["sxrange"] = [filteredSelection[0].x0, filteredSelection[0].x1].sort((a, b) => Date.parse(a) - Date.parse(b));
						obj["syrange"] = [filteredSelection[0].y0, filteredSelection[0].y1].sort(function (a, b) { return a - b }).map(roundTo10Decimals);
					}
					return obj
				});
			}
			console.log(filteredTriggerData);


			// check if autoscale/axis reset was triggered 
			// autorange is triggered per axis,
			// but showspikes only appears in relayout for xaxis and yaxis
			let notAutorange = true;
			let notShowspikes = true;
			filteredTriggerData.forEach(obj => {
				if (mainFigRelayout[`xaxis${obj.mainSubplotIndex + 1}.autorange`]
					&& mainFigRelayout[`yaxis${obj.mainSubplotIndex + 1}.autorange`]) {
					//use the coarse range as a reference for 
					//the range to be used in the relayout for the selection box
					obj.myrange = obj.cyrange;
					notAutorange = false;
					if (mainFigRelayout['xaxis.showspikes'] === false
						&& mainFigRelayout['xaxis.showspikes'] === false) {
						obj.mxrange = obj.cxrange;
						notShowspikes = false;
					}
				}
			});

			//i should put the full original currentSelections in the update, 
			//THEN only change the trigger ones?
			let updateCondition = false;
			update = { 'selections': currentSelections || [] }
			filteredTriggerData.forEach((obj, i) => {
				// console.log(!compareArrays(obj.sxrange,obj.mxrange))
				// console.log(!compareArrays(obj.syrange, obj.myrange));
				// console.log(notAutorange == notShowspikes);
				if ((!compareArrays(obj.sxrange, obj.mxrange) && notAutorange == notShowspikes) || !compareArrays(obj.syrange, obj.myrange)) {
					updateCondition = true;
					const columnIndex = +obj.columnIndex === 0 ? "" : String(obj.columnIndex + 1);
					const selectionIndex = update['selections'].findIndex(obj => obj.xref === `x${columnIndex}`);
					console.log(selectionIndex)

					update['selections'][selectionIndex] =
					{
						"xref": `x${columnIndex}`,
						"yref": `y${columnIndex}`,
						"line": {
							"width": 1,
							"dash": "dot"
						},
						"type": "rect",
						"x0": obj.mxrange[0],
						"x1": obj.mxrange[1],
						"y0": obj.myrange[0],
						"y1": obj.myrange[1]
					};
				}

			});
			// console.log(update);

			if (updateCondition) {
				// Update the layout without triggering another relayout event
				console.warn('main -> coarse');
				// document.dispatchEvent(); doesnt work
				Plotly.relayout(
					coarse_graphDiv, update
				);
			}

			return coarseFigID;
		},
		set_coarse_range: function (coarsefigure, mainFigID, coarseFigID, linkedIndices) {
			function getGraphDiv(gdID) {
				// see this link for more information https://stackoverflow.com/a/34002028 
				let graphDiv = document?.querySelectorAll('div[id*="' + gdID + '"][class*="dash-graph"]');
				if (graphDiv.length > 1) {
					throw new SyntaxError("UpdateStore: multiple graphs with ID=" + gdID + " found; n=" + graphDiv.length + " (either multiple graphs with same ID's or current ID is a str-subset of other graph IDs)");
				} else if (graphDiv.length < 1) {
					throw new SyntaxError("UpdateStore: no graphs with ID=" + gdID + " found");
				}
				graphDiv = graphDiv?.[0]?.getElementsByClassName('js-plotly-plot')?.[0];
				const isDOMElement = el => el instanceof HTMLElement
				if (!isDOMElement) {
					throw new Error(`Invalid gdID '${gdID}'`);
				}
				return graphDiv;
			}
			//can be changed for something more sophisticated if needed, 
			//it does the job of determining if two arrays have the same values for now
			const compareArrays = (a, b) => {
				return JSON.stringify(a) === JSON.stringify(b);
			};
			const roundTo10Decimals = (a) => {
				return +a.toFixed(10)
			};

			/**
			 * 
			 * @param {object} layout : layout object from which the x and y ranges should be extracted
			 * @param {Array} indexList : list of subplot indices of which the x and y range should be extracted. undefined = all subplots
			 * @returns an array containing an array of xranges and an array of yranges of the figure the layout comes from
			 */
			const getFigureInfo = (layout, indexList = undefined) => {
				const yrange = [];
				const xrange = [];


				Object.keys(layout).forEach(key => {
					if (key.startsWith("yaxis")) {
						const axisNumber = key.slice(5, 6) ? key.slice(5, 6) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							yrange.push({ [axisNumber]: axisRange });
						}
					} else if (key.startsWith("xaxis")) {
						const axisNumber = key.slice(5, 9) ? key.slice(5, 9) - 1 : 0; // Extract the axis number from the key
						if (!(indexList && !indexList.includes(axisNumber))) {
							const axisRange = layout[key].range;
							xrange.push({ [axisNumber]: axisRange });
						}
					}
				});
				return [xrange, yrange];
			};

			const getGridLayout = (layout) => {
				const xaxes = new Set();
				const yaxes = new Set();

				Object.keys(layout).forEach(key => {
					if (key.startsWith("yaxis")) {
						yaxes.add(JSON.stringify(layout[key].domain));
					} else if (key.startsWith("xaxis")) {
						xaxes.add(JSON.stringify(layout[key].domain));
					}
				});
				return [yaxes.size, xaxes.size]
			}
			let coarse_graphDiv = getGraphDiv(coarseFigID);
			let main_graphDiv = getGraphDiv(mainFigID);

			const [mainRows, mainCols] = getGridLayout(main_graphDiv.layout);

			// let linkedIndices = [1, 1, 2];
			if (mainCols > 0 && linkedIndices.length > mainCols) {
				linkedIndices = linkedIndices.slice(0, mainCols);
			} else if (mainCols === 0) {
				linkedIndices = [linkedIndices[0]];
			}

			linkedIndices = linkedIndices.map((item, i) => {
				if (mainRows === 0) {
					item = 0;
				} else if (item >= mainRows) {
					// set the row to the max possible index within the grid
					item = mainRows - 1;
				}
				return item;
			})

			
			// console.log(main_graphDiv._fullData);
			// console.log(coarse_graphDiv.data);
			let updateData = [];
			linkedIndices.forEach((item, i) => {
				const subplotIndex = +linkedIndices.length * item + i;
				const columnIndex = +i;
				// add color of linked plot traces to updateData => change trace colors of coarse graph
				let filteredTraceColors = main_graphDiv._fullData.filter(trace => {
					if (subplotIndex === 0) {
						return trace.xaxis === `x${subplotIndex + 1}` || trace.xaxis === "x"
					} else {
						return trace.xaxis === `x${subplotIndex + 1}`
					}

				}).map(obj => {
					return {
						"name": obj.name,
						"traceColor": obj.line.color
					}
				});

				const filteredTraceIndices = filteredTraceColors.map(obj => {
					return coarse_graphDiv.data.findIndex(trace => trace.name === obj.name);
				});

				updateData.push({
					"columnIndex": +i,
					"linkIndex": +item,
					"mainSubplotIndex": subplotIndex,
					"traceColors": filteredTraceColors.map(obj => obj.traceColor),
					"coarseTraceIndices": filteredTraceIndices
				})
			});


			console.log(updateData);

			const filteredTriggers = updateData.map(obj => obj.mainSubplotIndex);

			let [xrange, yrange] = getFigureInfo(main_graphDiv.layout, filteredTriggers);
			let [cxrange, cyrange] = getFigureInfo(coarse_graphDiv.layout);

			updateData = updateData.map(obj => {
				const subplotIndex = obj.mainSubplotIndex;
				const columnIndex = obj.columnIndex;
				const yrangeval = yrange.find(o => o.hasOwnProperty(subplotIndex));
				const xrangeval = xrange.find(o => o.hasOwnProperty(subplotIndex));
				const cyrangeval = cyrange.find(o => o.hasOwnProperty(columnIndex));
				const cxrangeval = cxrange.find(o => o.hasOwnProperty(columnIndex));

				if (xrangeval) {
					obj["mxrange"] = xrangeval[subplotIndex];
					obj["myrange"] = yrangeval[subplotIndex].map(roundTo10Decimals);
					obj["cxrange"] = cxrangeval[columnIndex];
					obj["cyrange"] = cyrangeval[columnIndex].map(roundTo10Decimals);
				}
				return obj
			});


			let updates = { 'selections': [] };
			let restyleUpdates = { "line.color": [] };
			let restyleTraces = [];
			updateData.forEach(obj => {
				const subplotIndex = +obj.columnIndex === 0 ? "" : String(obj.columnIndex + 1);
				if (!compareArrays(obj["mxrange"], obj["cxrange"]) || !compareArrays(obj["myrange"], obj["cyrange"])) {

					console.warn("updating coarse range");
					let xaxisKey0 = `xaxis${subplotIndex}.range[0]`;
					let xaxisKey1 = `xaxis${subplotIndex}.range[1]`;
					let yaxisKey0 = `yaxis${subplotIndex}.range[0]`;
					let yaxisKey1 = `yaxis${subplotIndex}.range[1]`;

					updates = {
						...updates,
						[xaxisKey0]: obj.mxrange[0],
						[xaxisKey1]: obj.mxrange[1],
						[yaxisKey0]: obj.myrange[0],
						[yaxisKey1]: obj.myrange[1],
					};
				}
				updates['selections'].push({
					"xref": `x${subplotIndex}`,
					"yref": `y${subplotIndex}`,
					"line": {
						"width": 1,
						"dash": "dot"
					},
					"type": "rect",
					"x0": obj.mxrange[0],
					"x1": obj.mxrange[1],
					"y0": obj.myrange[0],
					"y1": obj.myrange[1]
				});

				restyleUpdates['line.color'].push(...obj.traceColors);
				restyleTraces.push(...obj.coarseTraceIndices);

			});
			console.log(restyleUpdates);
			console.log(restyleTraces);




			// for (let i = 0; i < cxrange.length; i++) {

			// 	const xrange = Object.values(mxrange[i]);
			// 	const yrange = Object.values(myrange[i]);
			// 	if (!compareArrays(xrange, Object.values(cxrange[i])) || !compareArrays(yrange, Object.values(cyrange[i]))) {
			// 		console.warn("updating coarse range");

			// 		let xaxisKey0;
			// 		let xaxisKey1;
			// 		let yaxisKey0;
			// 		let yaxisKey1;

			// 		if (i == 0) {
			// 			xaxisKey0 = 'xaxis.range[0]';
			// 			xaxisKey1 = 'xaxis.range[1]';

			// 			yaxisKey0 = 'yaxis.range[0]';
			// 			yaxisKey1 = 'yaxis.range[1]';
			// 		} else {
			// 			xaxisKey0 = `xaxis${i + 1}.range[0]`;
			// 			xaxisKey1 = `xaxis${i + 1}.range[1]`;

			// 			yaxisKey0 = `yaxis${i + 1}.range[0]`;
			// 			yaxisKey1 = `yaxis${i + 1}.range[1]`;
			// 		}


			// 		updates[xaxisKey0] = xrange[0][0];
			// 		updates[xaxisKey1] = xrange[0][1];
			// 		updates[yaxisKey0] = yrange[0][0];
			// 		updates[yaxisKey1] = yrange[0][1];

			// 	}
			// 	updates['selections'].push({
			// 		"xref": `x${i + 1}`,
			// 		"yref": `y${i + 1}`,
			// 		"line": {
			// 			"width": 1,
			// 			"dash": "dot"
			// 		},
			// 		"type": "rect",
			// 		"x0": xrange[0][0],
			// 		"x1": xrange[0][1],
			// 		"y0": yrange[0][0],
			// 		"y1": yrange[0][1]
			// 	});


			// 	// let coarse_graphDiv2 = getGraphDiv(coarseFigID);
			// 	// let cyrange2 = coarse_graphDiv2.layout.yaxis.range;
			// 	// let cxrange2 = coarse_graphDiv2.layout.xaxis.range;
			// 	// console.log("coarse range x: " + cxrange2);
			// 	// console.log("coarse range y:" + cyrange2);
			// 	// console.log("main range x: " + mxrange);
			// 	// console.log("main range y: " + myrange);
			// 	
			// }
			console.log(updates);
			Plotly.relayout(coarse_graphDiv, updates);
			Plotly.restyle(coarse_graphDiv, restyleUpdates, restyleTraces);
			console.log(coarse_graphDiv.layout.selections);
			return mainFigID;
		}
	}
});