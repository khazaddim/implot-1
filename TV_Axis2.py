import dearcygui as dcg
import numpy as np
from datetime import datetime, timezone
from candle_utils.candle_gen import generate_fake_candlestick_data
from time_series_tools import find_gaps_and_chunks
import asyncio
from dearcygui.utils.asyncio_helpers import AsyncPoolExecutor, AsyncThreadPoolExecutor, run_viewport_loop
import time

from Demos.main_demo.text_utils import MarkDownText

# import my own candle utils
from DCG_Candle_Utils import PlotCandleStick

# To DO:
# 1. Implement dynamic axis label adjustment based on zoom level.
# 2. Highlight gaps and chunks in the candlestick data using the digital plot.
# 3. Determine if we use markdown or Text with spaces in the string of gaps and chunks

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop) # why did the AI recommend this?


class Candle_Testing:

    def _white_theme(self):
        viewport_theme = dcg.ThemeColorImGui(self.C,
            border_shadow=(0.960784375667572, 0.960784375667572, 0.960784375667572, 0.0),
            window_bg=(0.9490196704864502, 0.9058824181556702, 0.9058824181556702, 0.9411765336990356),
            title_bg=(0.9803922176361084, 0.9803922176361084, 0.9803922176361084, 1.0),
            text=(0.1, 0.9, 0.14509804546833038, 1.0))
        plot_theme = dcg.ThemeColorImPlot(self.C,
            axis_grid=(0.07450980693101883, 0.06666667014360428, 0.06666667014360428, 0.250980406999588)
        )
        theme = dcg.ThemeList(self.C)
        theme.children = [viewport_theme, plot_theme]
        return theme
   
    def __init__(self, white_theme=True):
        # create the context
        self.C = dcg.Context()
        self.C.queue = AsyncPoolExecutor()
        self.C.viewport.wait_for_input = True
        self.dense_label_state = False
        self.current_tab = ""

        # initialize the candle variables
        self.dates = np.array([])
        self.opens = np.array([])
        self.highs = np.array([])
        self.lows = np.array([])
        self.closes = np.array([])
        self.index = np.array([])
        self.volume = np.array([])

        self.orig_dates = np.array([])
        self.orig_opens = np.array([])
        self.orig_highs = np.array([])
        self.orig_lows = np.array([])
        self.orig_closes = np.array([])
        self.orig_index = np.array([])
        self.orig_volume = np.array([])

        # make a theme for the viewport
        self.viewport_theme = dcg.ThemeColorImGui(
            self.C,
            border_shadow=(0.960784375667572, 0.960784375667572, 0.960784375667572, 0.0),
            window_bg=(0.9490196704864502, 0.9058824181556702, 0.9058824181556702, 0.9411765336990356),
            title_bg=(0.9803922176361084, 0.9803922176361084, 0.9803922176361084, 1.0)
        )

        # Green digital theme
        with dcg.ThemeList(self.C) as self.digital_theme_green:
            dcg.ThemeColorImPlot(self.C, line=(0, 200, 0), fill=(0, 120, 0, 180))
            dcg.ThemeStyleImPlot(self.C, line_weight=2)

        # Red digital theme
        with dcg.ThemeList(self.C) as self.digital_theme_red:
            dcg.ThemeColorImPlot(self.C, line=(200, 0, 0), fill=(120, 0, 0, 180))
            dcg.ThemeStyleImPlot(self.C, line_weight=2)

        # Blue digital theme
        with dcg.ThemeList(self.C) as self.digital_theme_blue:
            dcg.ThemeColorImPlot(self.C, line=(0, 0, 200), fill=(0, 75, 120, 120))
            dcg.ThemeStyleImPlot(self.C, line_weight=0.1)


        # initialize the viewport with the theme
        if white_theme:
            self.C.viewport.initialize(height=900, width=1600, theme=self._white_theme())  # , resizable=False
        else:
            self.C.viewport.initialize(height=900, width=1600) #, wait_for_input=True)

        # create the window
        with dcg.Window(self.C, label="plot prototype", primary=True, width='fillx', height='filly') as win:
            # make a menu bar
            with dcg.MenuBar(self.C):
                with dcg.Menu(self.C, label="File"):
                    dcg.MenuItem(self.C, label="New", callback=lambda s, t, d: print("New file"))
                    dcg.MenuItem(self.C, label="Open", callback=lambda s, t, d: print("Open file"))
                    dcg.MenuItem(self.C, label="Save", callback=lambda s, t, d: print("Save file"))
                with dcg.Menu(self.C, label="Settings"):
                    dcg.MenuItem(self.C, label="Open Style Editor", callback=lambda s, t, d: dcg.utils.StyleEditor(self.C))
            # create a horizontal layout to hold the resizable window with a default width and the other with just a regular child window
            with dcg.HorizontalLayout(self.C,no_wrap=True):
                with dcg.ChildWindow(self.C, label="Left Side", width=300, resizable_x=True) as self.left_win:
                    # for use later, i think for if i have lots of complex objects:
                    # ,attach=False will allow one to make an object that is not attached to the parent layout
                    # you can then set the parent attribute of the object later
                    # this means that they will not be attached to the rendering tree until you set the parent attribute
                    self.plot_button = dcg.Button(
                        self.C,
                        label="Plot Candle Data",
                        width='fillx', height='win.height/24',
                        callback=lambda s, a, u: self.plot_candle_data(s, a, u, plot=self.plot),
                        user_data='sneaky user data'  # I can retrieve this in the callback with target.user_data. cool!
                    )
                    # add controls for fetching data

                    # gaps n' chunks button
                    self.gaps_button = dcg.Button(
                        self.C,
                        label="Gaps n' Chunks",
                        width='fillx', height='win.height/24',
                        callback=self.add_gaps_chunks_GUI
                    )
                    self.Collapse_time_button = dcg.Button(
                        self.C,
                        label="Collapse Time",
                        width='fillx', height='win.height/24',
                        callback=self.collapse_time_chart
                    )
                    self.Collapse_time_vec_button = dcg.Button(
                        self.C,
                        label="Collapse Time Vec",
                        width='fillx', height='win.height/24',
                        callback=self.collapse_time_chart_vec
                    )
                    # launch second window button
                    self.launch_button = dcg.Button(
                        self.C,
                        label="Launch Second Window",
                        width='fillx', height='win.height/24',
                        #callback=lambda s, a, u: asyncio.run_coroutine_threadsafe(
                        #    self.create_custom_viewport(), loop
                        #)
                        callback=lambda: self.create_viewport_simple()  # create_custom_viewport
                    )
                    self.five_min_label = dcg.Text(self.C,
                                                   value="This is a sample text that will demonstrate text wrapping functionality in Dear CyGui.\n Resize the window to see how the text wraps accordingly.",
                                                   wrap=300, height='filly'
                                                   )
                    # self.five_min_label = MarkDownText(self.C, value="5 Min Data") #, wrap=300)
                with dcg.ChildWindow(self.C, label="Right Side") as self.right_win:
                    with dcg.TabBar(self.C):
                        with dcg.Tab(self.C, label="Colapsed Time Chart") as self.Collapsed_time_tab:
                            self.plot=dcg.Plot(self.C, label="Test Plot", width='fillx', height='filly', has_box_select=True)
                            self.plot_candle_data(None, None, None, plot=self.plot)


                        with dcg.Tab(self.C, label="Original Time Chart") as self.Original_time_tab:
                            with dcg.Plot(self.C, label="Original Time Plot", width='fillx', height='filly', has_box_select=True) as self.orig_plot:
                                # Configure plot for time series
                                # , width='win.width/1.8', height='win.height/1.2'
                                # , width='fillx', height='filly'
                                # Set axis labels and scales
                                self.orig_plot.X1.label = "Date"
                                self.orig_plot.X1.scale = dcg.AxisScale.TIME  # this does not seem to be needed with the custom axis
                                self.orig_plot.Y1.label = "Price ($)"
                                # Generate fake candlestick data
                                self.orig_dates, self.orig_opens, self.orig_highs, self.orig_lows, self.orig_closes, self.orig_index, _  = generate_fake_candlestick_data(
                                    remove_weekends=False, interval='hourly', length=300
                                )

                                # Use DearCyGui's utility function for candlestick charts
                                self.orig_candlestick_plot = PlotCandleStick(
                                    self.C,
                                    dates=self.orig_dates,
                                    opens=self.orig_opens,
                                    closes=self.orig_closes,
                                    lows=self.orig_lows,
                                    highs=self.orig_highs,
                                    label="Stock Price",
                                    weight=0.1,
                                    time_formatter=lambda x: datetime.fromtimestamp(x).strftime("%b %d")
                                )


        # add the callback to the handler list in the plot object
        self.plot.handlers += [
            #dcg.MouseCursorHandler(C, callback=mouse_click_callback),
            dcg.AxesResizeHandler(self.C, callback=self.axes_resize_callback)
        ]
        self.Collapsed_time_tab.handlers += [
            dcg.ActiveHandler(self.C, callback=lambda s, a, u: self.set_Current_Tab(s, a, u))
        ]
        self.Original_time_tab.handlers += [
            dcg.ActiveHandler(self.C, callback=lambda s, a, u: self.set_Current_Tab(s, a, u))
        ]
        self.left_win.handlers += [
            dcg.ResizeHandler(self.C, callback=self.on_resize)
        ]

    def on_resize(self, sender, app_data):
        # print the new size for debugging
        print(f"five_min_label resized to: {app_data.width.value}")
        self.five_min_label.wrap = app_data.width.value - 80  # Adjust for padding

    def axes_resize_callback(self, sender, target, data):
        """
        Callback for AxesResizeHandler. Prints diagnostics and updates X axis labels based on scaling.
        Only updates labels if the dense label state changes.
        """
        debug_print = False
        if debug_print:
            print(f"[AxesResizeHandler] Callback triggered. Data: {data}")
        x_axis_info = data[0]  # (min_time, max_time, scaling_factor)
        min_time, max_time, scaling_factor = x_axis_info
        if debug_print:
            print(f"[AxesResizeHandler] X axis min_time: {min_time}, max_time: {max_time}, scaling_factor: {scaling_factor}")

        # Determine if we should be in dense label mode
        should_be_dense = scaling_factor > 200

        if should_be_dense != self.dense_label_state:
            # State changed, update labels
            self.dense_label_state = should_be_dense
            if should_be_dense:
                labels = [str(i)+'\n'+'d' for idx, i in enumerate(self.index) if idx % 2 == 0]
                coords = self.dates[::2]
                # no_gridlines = True
                if debug_print:
                    print(f"[AxesResizeHandler] Entering dense label state, reducing labels to {len(labels)} entries.")
            else:
                labels = [str(i)+'\n'+'s' for i in self.index]
                coords = self.dates
                # no_gridlines = False
                if debug_print:
                    print(f"[AxesResizeHandler] Exiting dense label state, using all labels ({len(labels)} entries).")

            self.plot.X1.labels = labels
            self.plot.X1.labels_coord = coords
            self.plot.X1.no_gridlines = False
            self.plot.X1.labels_major = [False, True] * (len(labels) // 2)  # Make all labels major for visibility
        else:
            if debug_print:
                print(f"[AxesResizeHandler] Dense label state unchanged ({self.dense_label_state}), no label update.")

    def add_gaps_chunks_GUI(self, sender, app_data, user_data):
        """
        GUI callback to find and print gaps and chunks in the candlestick data.
        Prints each dict key on its own line and labels entries like "Gap 1", "Chunk 2", etc.
        """
        gaps_n_chunks = find_gaps_and_chunks(self.dates)

        if not gaps_n_chunks:
            self.five_min_label.value = "No gaps or chunks found."
            return

        # simple per-type counters
        counts = {}

        def detect_label(obj):
            """Return 'Gap', 'Chunk' or fallback 'Item' based on contents."""
            try:
                # prefer explicit type key
                if isinstance(obj, dict):
                    for k in obj:
                        if str(k).lower() == "type":
                            v = str(obj[k]).lower()
                            if "gap" in v:
                                return "Gap"
                            if "chunk" in v:
                                return "Chunk"
                # search keys/values for hints
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        ks = str(k).lower()
                        vs = str(v).lower()
                        if "gap" in ks or "gap" in vs:
                            return "Gap"
                        if "chunk" in ks or "chunk" in vs:
                            return "Chunk"
                else:
                    s = str(obj).lower()
                    if "gap" in s:
                        return "Gap"
                    if "chunk" in s:
                        return "Chunk"
            except Exception:
                pass
            return "Item"

        lines = []
        for gc in gaps_n_chunks:
            label = detect_label(gc)
            counts[label] = counts.get(label, 0) + 1
            lines.append(f"{label} {counts[label]}:")

            if isinstance(gc, dict):
                # print each key/value on its own line
                for k, v in gc.items():
                    lines.append(f"  {k}: {v}")
            else:
                # try to handle mapping-like objects, otherwise fall back to str()
                try:
                    items = getattr(gc, "items", None)
                    if callable(items):
                        for k, v in items():
                            lines.append(f"  {k}: {v}")
                    else:
                        lines.append(f"  {gc}")
                except Exception:
                    lines.append(f"  {gc}")

            lines.append("")  # blank line between entries

        str_out = "Gaps and Chunks found:\n" + "\n".join(lines)
        print(gaps_n_chunks)
        self.five_min_label.value = str_out

    def collapse_time_chart(self, sender, app_data, user_data):
        # get the starting time
        start_time = time.time()
        """
        Callback to collapse time in the candlestick chart when the button is pressed.
        """
        # call gaps and chunks function
        gaps_n_chunks = find_gaps_and_chunks(self.dates)
        print(gaps_n_chunks)
        # for each gap in the list, subtract the gap size from all subsequent dates
        collapsed_dates = self.dates.copy()
        cumulative_shift = 0
        for item in gaps_n_chunks:
            if item.get("Type", "").lower() == "gap":
                start = item.get("start", None)
                stop = item.get("stop", None)
                print(f"Found gap from {start} to {stop}.")
                if start is not None and stop is not None:
                    gap_size = stop - start
                    collapsed_dates = np.array([date - gap_size if date + cumulative_shift > start else date for date in collapsed_dates])
                    cumulative_shift += gap_size
                    print(f"Collapsed gap from {start} to {stop}, size {gap_size} seconds.")
            else:
                print(f"Did not find gap type in item: {item}")

        # store a copy of the old dates for a safe comparison
        old_dates = np.array(self.candlestick_plot.dates, copy=True)

        self.dates = collapsed_dates
        self.candlestick_plot.dates = self.dates # validate and render called internally
        #verify that the dates have changed
        if np.array_equal(old_dates, self.candlestick_plot.dates):
            print("Dates did not change after collapsing time.")
        else:
            print("Dates successfully collapsed.")
        # apply the collapsed dates to the candlestick plot
        
        end_time = time.time()
        self.five_min_label.value =  f"Time collapse took {end_time - start_time:.4f} seconds."

    def collapse_time_chart_vec(self, sender, app_data, user_data):
        """
        Callback to collapse time in the candlestick chart when the button is pressed.
        """
        start_time = time.time()
        gaps_n_chunks = find_gaps_and_chunks(self.dates)
        print(gaps_n_chunks)

        # work from an immutable copy of the original timeline
        orig_dates = np.array(self.dates, copy=True, dtype=float)
        # per-element cumulative shift (same shape as orig_dates)
        total_shift = np.zeros_like(orig_dates, dtype=float)

        # sort gaps by start time so behavior is deterministic
        gaps_sorted = sorted(gaps_n_chunks, key=lambda it: it.get("start", float("inf")))

        for item in gaps_sorted:
            # accept either 'type' or 'Type'
            itype = str(item.get("type", item.get("Type", ""))).lower()
            if itype != "gap":
                print(f"Skipping non-gap item: {item}")
                continue

            start = item.get("start", None)
            stop = item.get("stop", None)
            if start is None or stop is None:
                print(f"Skipping malformed gap item: {item}")
                continue

            gap_size = stop - start
            # mark all original timestamps strictly after the gap start
            mask = orig_dates > start
            total_shift[mask] += gap_size
            print(f"Will collapse gap from {start} to {stop}, size {gap_size} seconds.")

        # apply all shifts once (vectorized)
        collapsed_dates = orig_dates - total_shift

        # update via the public API so validation + render happen
        old_dates = np.array(self.candlestick_plot.dates, copy=True)
        self.candlestick_plot.update(dates=collapsed_dates)
        # keep the local copy in sync
        self.dates = collapsed_dates

        if np.array_equal(old_dates, np.array(self.candlestick_plot.dates)):
            print("Dates did not change after collapsing time.")
        else:
            print("Dates successfully collapsed.")

        end_time = time.time()
        self.five_min_label.value = f"Time collapse took {end_time - start_time:.4f} seconds."
        

    def plot_candle_data(self, sender, target, user_data, plot=None):
        if target is not None:
            print(target.user_data) # I could use this to get user_data
        with plot:
            plot.X1.label = "Date"
            plot.Y1.label = "Price ($)"
            self.dates, self.opens, self.highs, self.lows, self.closes, self.index, self.volume = generate_fake_candlestick_data(
                remove_weekends=True, interval='hourly', length=500
            )
            # see if there are any plotted items already, if so, delete them
            # existing_items = plot.get_items() # not sure how to do this
            # make my custom candlestick plot object
            # see if self.candlestick_plot exists already
            if hasattr(self, 'candlestick_plot'):
                # update existing plot
                print("Updating existing candlestick plot.")
                self.candlestick_plot.update_all(
                    dates=self.dates,
                    opens=self.opens,
                    closes=self.closes,
                    lows=self.lows,
                    highs=self.highs,
                    volumes=self.volume
                )
            else:
                # create new plot
                # for now i am just using self.candlestick_plot but i will want to make this more flexible later
                # I could have a dictionary of plots that have different parents
                # this function could be used to update them based on their target.user_data
                # the button user_data would specify which plot to update
                # or it could be passed in as a parameter in the lambda function callback when creating the button
                # that would be cleaner
                print("Creating new candlestick plot.")
                self.candlestick_plot = PlotCandleStick( 
                    self.C,
                    dates=self.dates,
                    opens=self.opens,
                    closes=self.closes,
                    lows=self.lows,
                    highs=self.highs,
                    volumes=self.volume,
                    volume_kwargs={'theme':self.digital_theme_green},
                    label="Stock Price",
                    weight=0.1,
                    time_formatter=lambda x: datetime.fromtimestamp(x).strftime("%b %d")
                )

                #stuff for plotting custom axis labels
                self.plot.X1.keep_default_ticks = False
                self.plot.X1.labels = [str(i) for i in self.index]  # Use index as labels
                self.plot.X1.labels_coord = self.dates
                self.plot.X1.no_gridlines = [False]*len(self.index)

                # this plot needs the context modifier
                '''dcg.PlotDigital(
                    self.C,
                    X=self.dates,
                    Y=self.volume,
                    label="Volume",
                    theme=self.digital_theme_green
                )'''


    def set_Current_Tab(self, sender=None, app_data=None, user_data=None):
        """
        Utility to get the value of each tab to see which is active.
        """
        #print the passed parameters
        #print(f"Sender: {sender}, App Data: {app_data}, User Data: {user_data}")
        print(app_data.label)
        self.current_tab = app_data.label

    # used for non-async version
    def run_viewport_loop(self):
        """
        Run the viewport loop until it is closed.
        """
        while self.C.running:
            self.C.viewport.render_frame()


    async def create_viewport_simple(self):
        # Create a new viewport context
        try:
            self.new_context = dcg.Context()
            if self._white_theme:
                self.new_context.viewport.initialize(width=400, height=300, title="New Viewport", theme=self._white_theme())
            else:
                self.new_context.viewport.initialize(width=400, height=300, title="New Viewport")
        except Exception as e:
            with dcg.Window(self.new_context, modal=True):
                dcg.Text(self.new_context, value=f"Error creating new viewport: {e}")
            return
       
        # Add some items to the new viewport
        with dcg.Window(self.new_context, primary=True):
            dcg.Text(self.new_context, value="This is a new viewport!", x=10, y=10)
            # add a test button
            dcg.Button(self.new_context, label="Test Button", x=10, y=50)

        await run_viewport_loop(self.new_context.viewport)

        # Properly clean up the viewport
        self.new_context.queue.shutdown()
        self.new_context.viewport.delete_item()
        self.new_context.viewport.destroy()

# create an instance of the class and run the viewport loop
# non-async version
'''if __name__ == "__main__":
    candle_test = Candle_Testing(white_theme=False)
    candle_test.run_viewport_loop()'''

# async version
if __name__ == "__main__":
    app = Candle_Testing(white_theme=False)
    try:
        loop.run_until_complete(run_viewport_loop(app.C.viewport))
    finally:
        #app.process_pool.shutdown()
        print("No process pool to clean up.")